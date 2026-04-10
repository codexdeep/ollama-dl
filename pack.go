// Package pack handles creating and extracting portable .tar.zst model archives.
package pack

import (
	"archive/tar"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"

	"github.com/user/ollama-dl/internal/registry"
)

const (
	manifestEntry = "manifest.json"
	blobsPrefix   = "blobs/"
	magicComment  = "ollama-dl archive v1"
)

// PackOptions configures archive creation.
type PackOptions struct {
	// CompressionLevel: 1 (fast) to 22 (best). Default 3.
	CompressionLevel int
}

// Pack creates a .tar.zst archive at destPath containing:
//   - manifest.json
//   - blobs/<sha256-digest> for every layer
//
// sourceBlobsDir is the directory containing the blob files.
func Pack(manifest *registry.Manifest, rawManifest []byte, sourceBlobsDir string, destPath string, opts PackOptions) error {
	if opts.CompressionLevel == 0 {
		opts.CompressionLevel = 3
	}

	f, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("creating archive: %w", err)
	}
	defer f.Close()

	level, err := zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(opts.CompressionLevel))
	if err != nil {
		return fmt.Errorf("zstd level: %w", err)
	}
	zw, err := zstd.NewWriter(f, level)
	if err != nil {
		return fmt.Errorf("creating zstd writer: %w", err)
	}
	defer zw.Close()

	tw := tar.NewWriter(zw)
	defer tw.Close()

	// Write manifest
	if err := writeBytesToTar(tw, manifestEntry, rawManifest); err != nil {
		return fmt.Errorf("writing manifest to archive: %w", err)
	}

	// Write each blob
	for _, layer := range manifest.AllLayers() {
		filename := registry.DigestToFilename(layer.Digest)
		srcPath := filepath.Join(sourceBlobsDir, filename)

		info, err := os.Stat(srcPath)
		if err != nil {
			return fmt.Errorf("blob %s not found in %s: %w", filename, sourceBlobsDir, err)
		}

		if err := writeFileToTar(tw, blobsPrefix+filename, srcPath, info); err != nil {
			return fmt.Errorf("archiving blob %s: %w", filename, err)
		}

		fmt.Printf("  + %-60s  %s\n", filename, humanSize(info.Size()))
	}

	return nil
}

// Unpack extracts a .tar.zst archive into destDir.
// Returns the raw manifest bytes found in the archive.
func Unpack(archivePath string, destDir string) ([]byte, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, fmt.Errorf("opening archive: %w", err)
	}
	defer f.Close()

	zr, err := zstd.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("creating zstd reader: %w", err)
	}
	defer zr.Close()

	tr := tar.NewReader(zr)

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return nil, err
	}

	var rawManifest []byte

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading archive: %w", err)
		}

		// Security: prevent path traversal
		if strings.Contains(hdr.Name, "..") {
			return nil, fmt.Errorf("unsafe path in archive: %s", hdr.Name)
		}

		destPath := filepath.Join(destDir, filepath.FromSlash(hdr.Name))

		if hdr.Typeflag == tar.TypeDir {
			os.MkdirAll(destPath, 0o755)
			continue
		}

		// Ensure parent dir exists
		if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
			return nil, err
		}

		if hdr.Name == manifestEntry {
			data, err := io.ReadAll(tr)
			if err != nil {
				return nil, fmt.Errorf("reading manifest from archive: %w", err)
			}
			rawManifest = data
			// Also write to disk
			if err := os.WriteFile(destPath, data, 0o644); err != nil {
				return nil, err
			}
			continue
		}

		outF, err := os.Create(destPath)
		if err != nil {
			return nil, fmt.Errorf("creating %s: %w", destPath, err)
		}

		n, err := io.Copy(outF, tr)
		outF.Close()
		if err != nil {
			os.Remove(destPath)
			return nil, fmt.Errorf("extracting %s: %w", hdr.Name, err)
		}
		fmt.Printf("  ✓ %-60s  %s\n", filepath.Base(hdr.Name), humanSize(n))
	}

	if rawManifest == nil {
		return nil, fmt.Errorf("archive does not contain manifest.json")
	}
	return rawManifest, nil
}

// Inspect reads and prints metadata from an archive without extracting it.
func Inspect(archivePath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	stat, _ := f.Stat()

	zr, err := zstd.NewReader(f)
	if err != nil {
		return err
	}
	defer zr.Close()

	tr := tar.NewReader(zr)

	fmt.Printf("Archive : %s\n", archivePath)
	fmt.Printf("Size    : %s\n\n", humanSize(stat.Size()))

	type entry struct {
		name string
		size int64
	}
	var entries []entry
	var manifestBytes []byte

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		data, _ := io.ReadAll(tr)
		entries = append(entries, entry{hdr.Name, int64(len(data))})
		if hdr.Name == manifestEntry {
			manifestBytes = data
		}
	}

	if manifestBytes != nil {
		var m registry.Manifest
		if json.Unmarshal(manifestBytes, &m) == nil {
			fmt.Printf("Layers  : %d\n\n", len(m.AllLayers()))
		}
	}

	fmt.Println("Contents:")
	for _, e := range entries {
		fmt.Printf("  %-64s  %s\n", e.name, humanSize(e.size))
	}
	return nil
}

// ArchiveName returns a sanitized filename for a model archive.
func ArchiveName(ref registry.ModelRef) string {
	name := fmt.Sprintf("%s_%s_%s_%s.tar.zst",
		ref.Namespace, ref.Repo, ref.Tag,
		time.Now().Format("20060102"),
	)
	return strings.ReplaceAll(name, "/", "_")
}

// -- helpers -----------------------------------------------------------------

func writeBytesToTar(tw *tar.Writer, name string, data []byte) error {
	hdr := &tar.Header{
		Name:    name,
		Mode:    0o644,
		Size:    int64(len(data)),
		ModTime: time.Now(),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	_, err := tw.Write(data)
	return err
}

func writeFileToTar(tw *tar.Writer, name string, srcPath string, info os.FileInfo) error {
	hdr := &tar.Header{
		Name:    name,
		Mode:    0o644,
		Size:    info.Size(),
		ModTime: info.ModTime(),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	f, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(tw, f)
	return err
}

func humanSize(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}
