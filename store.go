// Package ollama handles reading/writing the local Ollama models directory.
package ollama

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/user/ollama-dl/internal/registry"
)

// Store represents the local Ollama models directory.
type Store struct {
	Root string
}

// NewStore creates a Store, auto-detecting the root if dir is empty.
func NewStore(dir string) (*Store, error) {
	if dir == "" {
		var err error
		dir, err = DetectDir()
		if err != nil {
			return nil, err
		}
	}
	return &Store{Root: dir}, nil
}

// BlobsDir returns the blobs sub-directory path.
func (s *Store) BlobsDir() string { return filepath.Join(s.Root, "blobs") }

// ManifestsDir returns the manifests sub-directory path.
func (s *Store) ManifestsDir() string { return filepath.Join(s.Root, "manifests") }

// BlobPath returns the expected path for a given digest.
func (s *Store) BlobPath(digest string) string {
	return filepath.Join(s.BlobsDir(), registry.DigestToFilename(digest))
}

// ManifestPath returns the path for a model ref's manifest file.
func (s *Store) ManifestPath(ref registry.ModelRef) string {
	return filepath.Join(s.ManifestsDir(), "registry.ollama.ai", ref.Namespace, ref.Repo, ref.Tag)
}

// WriteManifest writes raw manifest JSON to the correct location for ref.
func (s *Store) WriteManifest(ref registry.ModelRef, data []byte) error {
	dir := filepath.Dir(s.ManifestPath(ref))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(s.ManifestPath(ref), data, 0o644)
}

// ReadManifest reads and parses the manifest for a locally installed model.
func (s *Store) ReadManifest(ref registry.ModelRef) (*registry.Manifest, []byte, error) {
	path := s.ManifestPath(ref)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("manifest not found at %s: %w", path, err)
	}
	var m registry.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, nil, fmt.Errorf("parsing manifest: %w", err)
	}
	return &m, data, nil
}

// InstalledModels walks the manifests directory and returns all installed model refs.
func (s *Store) InstalledModels() ([]registry.ModelRef, error) {
	base := filepath.Join(s.ManifestsDir(), "registry.ollama.ai")
	var refs []registry.ModelRef

	err := filepath.Walk(base, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(base, path)
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) != 3 {
			return nil
		}
		refs = append(refs, registry.ModelRef{
			Namespace: parts[0],
			Repo:      parts[1],
			Tag:       parts[2],
			Base:      registry.DefaultBase,
		})
		return nil
	})
	return refs, err
}

// VerifyResult holds the result of verifying a single blob.
type VerifyResult struct {
	Digest   string
	Filename string
	Size     int64
	Status   string // "ok", "missing", "size_mismatch", "hash_mismatch"
	Detail   string
}

// VerifyBlob checks whether a blob on disk matches its expected digest and size.
func (s *Store) VerifyBlob(layer registry.Layer) VerifyResult {
	filename := registry.DigestToFilename(layer.Digest)
	path := filepath.Join(s.BlobsDir(), filename)

	res := VerifyResult{Digest: layer.Digest, Filename: filename, Size: layer.Size}

	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		res.Status = "missing"
		res.Detail = "file not found"
		return res
	}
	if err != nil {
		res.Status = "missing"
		res.Detail = err.Error()
		return res
	}

	if info.Size() != layer.Size {
		res.Status = "size_mismatch"
		res.Detail = fmt.Sprintf("expected %d bytes, got %d", layer.Size, info.Size())
		return res
	}

	// Compute SHA256
	f, err := os.Open(path)
	if err != nil {
		res.Status = "missing"
		res.Detail = err.Error()
		return res
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		res.Status = "hash_mismatch"
		res.Detail = fmt.Sprintf("hashing error: %v", err)
		return res
	}

	got := "sha256:" + hex.EncodeToString(h.Sum(nil))
	if got != layer.Digest {
		res.Status = "hash_mismatch"
		res.Detail = fmt.Sprintf("expected %s, got %s", layer.Digest, got)
		return res
	}

	res.Status = "ok"
	return res
}

// DetectDir returns the default Ollama models directory for the current OS.
func DetectDir() (string, error) {
	if d := os.Getenv("OLLAMA_MODELS"); d != "" {
		return d, nil
	}
	switch runtime.GOOS {
	case "linux", "darwin":
		return filepath.Join(os.Getenv("HOME"), ".ollama", "models"), nil
	case "windows":
		return filepath.Join(os.Getenv("USERPROFILE"), ".ollama", "models"), nil
	default:
		return "", fmt.Errorf("unsupported OS %q — use --ollama-dir", runtime.GOOS)
	}
}
