// Package cmd contains all CLI command implementations.
package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/user/ollama-dl/internal/download"
	"github.com/user/ollama-dl/internal/ollama"
	"github.com/user/ollama-dl/internal/pack"
	"github.com/user/ollama-dl/internal/registry"
)

// Root returns the top-level cobra command.
func Root() *cobra.Command {
	root := &cobra.Command{
		Use:   "ollama-dl",
		Short: "Advanced Ollama model manager",
		Long: `ollama-dl is an advanced CLI for managing Ollama models.

Features:
  urls     — get direct download URLs for a model's blobs
  pull     — download blobs directly with parallel resumable downloads + progress bars
  install  — install downloaded blobs into the local Ollama store
  info     — inspect a model's architecture, family, and parameters
  diff     — compare two model versions blob-by-blob
  verify   — SHA256-check installed model blobs for corruption
  search   — search the Ollama library for available models
  pack     — bundle a model into a portable .tar.zst archive
  unpack   — extract and install a packed archive
  batch    — download multiple models from a YAML/JSON manifest`,
	}

	root.AddCommand(
		urlsCmd(),
		pullCmd(),
		installCmd(),
		infoCmd(),
		diffCmd(),
		verifyCmd(),
		searchCmd(),
		packCmd(),
		unpackCmd(),
		batchCmd(),
	)
	return root
}

// ---------------------------------------------------------------------------
// urls
// ---------------------------------------------------------------------------

func urlsCmd() *cobra.Command {
	var jsonOutput bool
	var registryBase string

	cmd := &cobra.Command{
		Use:   "urls <model>",
		Short: "Print direct download URLs for all model blobs",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ref, err := registry.ParseModelRef(args[0], registryBase)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "→ Fetching manifest for %s\n", ref)
			manifest, rawManifest, err := registry.FetchManifest(ref)
			if err != nil {
				return err
			}

			type BlobInfo struct {
				Filename  string `json:"filename"`
				Digest    string `json:"digest"`
				Size      int64  `json:"size"`
				MediaType string `json:"media_type"`
				URL       string `json:"url"`
			}

			var blobs []BlobInfo
			total := int64(0)
			for _, l := range manifest.AllLayers() {
				blobs = append(blobs, BlobInfo{
					Filename:  registry.DigestToFilename(l.Digest),
					Digest:    l.Digest,
					Size:      l.Size,
					MediaType: l.MediaType,
					URL:       ref.BlobURL(l.Digest),
				})
				total += l.Size
			}

			if jsonOutput {
				return json.NewEncoder(os.Stdout).Encode(map[string]any{
					"model": ref.String(), "blobs": blobs,
					"manifest_json": string(rawManifest),
				})
			}

			fmt.Printf("\nModel : %s\nBlobs : %d  Total: %s\n\n", ref, len(blobs), humanSize(total))
			for i, b := range blobs {
				fmt.Printf("── %d/%d  %s\n", i+1, len(blobs), b.MediaType)
				fmt.Printf("   File : %s  (%s)\n", b.Filename, humanSize(b.Size))
				fmt.Printf("   URL  : %s\n\n", b.URL)
			}

			fmt.Println("── manifest.json ──────────────────────────────────")
			fmt.Println(string(rawManifest))
			fmt.Println("\n── wget commands ──────────────────────────────────")
			for _, b := range blobs {
				fmt.Printf("wget -c -O %s '%s'\n", b.Filename, b.URL)
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOutput, "json", false, "JSON output")
	cmd.Flags().StringVar(&registryBase, "registry", "", "Override registry base URL")
	return cmd
}

// ---------------------------------------------------------------------------
// pull  (built-in downloader)
// ---------------------------------------------------------------------------

func pullCmd() *cobra.Command {
	var (
		outDir       string
		concurrency  int
		registryBase string
		retries      int
	)

	cmd := &cobra.Command{
		Use:   "pull <model>",
		Short: "Download a model's blobs directly with resumable parallel downloads",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ref, err := registry.ParseModelRef(args[0], registryBase)
			if err != nil {
				return err
			}

			if outDir == "" {
				outDir = sanitizeDirName(ref.Repo + "-" + ref.Tag)
			}
			if err := os.MkdirAll(outDir, 0o755); err != nil {
				return err
			}

			fmt.Fprintf(os.Stderr, "→ Fetching manifest for %s\n", ref)
			manifest, rawManifest, err := registry.FetchManifest(ref)
			if err != nil {
				return err
			}

			// Save manifest
			manifestPath := filepath.Join(outDir, "manifest.json")
			if err := os.WriteFile(manifestPath, rawManifest, 0o644); err != nil {
				return fmt.Errorf("saving manifest: %w", err)
			}
			fmt.Printf("✓ Manifest saved → %s\n\n", manifestPath)

			jobs := download.BuildJobs(ref, manifest, outDir)
			if len(jobs) == 0 {
				fmt.Println("✅ All blobs already downloaded.")
				return nil
			}

			results := download.Run(context.Background(), jobs, download.Options{
				Concurrency: concurrency,
				RetryMax:    retries,
			})

			fmt.Println()
			failed := 0
			for _, r := range results {
				if r.Err != nil {
					fmt.Fprintf(os.Stderr, "✗ %s: %v\n", r.Job.Label, r.Err)
					failed++
				}
			}
			if failed > 0 {
				return fmt.Errorf("%d blob(s) failed to download", failed)
			}
			fmt.Printf("✅ All blobs downloaded to %s\n", outDir)
			fmt.Printf("   Install with: ollama-dl install %s <model-name>\n", outDir)
			return nil
		},
	}

	cmd.Flags().StringVarP(&outDir, "out", "o", "", "Output directory (default: <repo>-<tag>)")
	cmd.Flags().IntVarP(&concurrency, "concurrency", "j", 3, "Parallel downloads")
	cmd.Flags().IntVar(&retries, "retries", 3, "Max retries per blob")
	cmd.Flags().StringVar(&registryBase, "registry", "", "Override registry base URL")
	return cmd
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

func installCmd() *cobra.Command {
	var (
		ollamaDir string
		symlink   bool
	)

	cmd := &cobra.Command{
		Use:   "install <source-dir> <model-name>",
		Short: "Install downloaded blobs into the local Ollama store",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			sourceDir, modelName := args[0], args[1]

			store, err := ollama.NewStore(ollamaDir)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "→ Ollama store: %s\n", store.Root)

			manifestData, err := os.ReadFile(filepath.Join(sourceDir, "manifest.json"))
			if err != nil {
				return fmt.Errorf("reading manifest.json: %w", err)
			}
			var manifest registry.Manifest
			if err := json.Unmarshal(manifestData, &manifest); err != nil {
				return err
			}

			ref, err := registry.ParseModelRef(modelName, "")
			if err != nil {
				return err
			}

			if err := os.MkdirAll(store.BlobsDir(), 0o755); err != nil {
				return err
			}

			fmt.Printf("\nInstalling %d blob(s)...\n", len(manifest.AllLayers()))
			for _, layer := range manifest.AllLayers() {
				if err := installBlob(store, sourceDir, layer, symlink); err != nil {
					return err
				}
			}

			if err := store.WriteManifest(ref, manifestData); err != nil {
				return fmt.Errorf("writing manifest: %w", err)
			}
			fmt.Printf("✓ Manifest registered\n")
			fmt.Printf("\n✅ %s installed!\n", ref)
			if ref.Namespace == "library" {
				fmt.Printf("   Run: ollama run %s:%s\n", ref.Repo, ref.Tag)
			} else {
				fmt.Printf("   Run: ollama run %s\n", ref)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&ollamaDir, "ollama-dir", "", "Override Ollama models directory")
	cmd.Flags().BoolVar(&symlink, "symlink", false, "Symlink blobs instead of copying (saves disk space)")
	return cmd
}

func installBlob(store *ollama.Store, sourceDir string, layer registry.Layer, useSymlink bool) error {
	filename := registry.DigestToFilename(layer.Digest)
	srcPath := filepath.Join(sourceDir, filename)

	// Also try blobs/ subdirectory (unpack puts blobs there)
	if _, err := os.Stat(srcPath); os.IsNotExist(err) {
		alt := filepath.Join(sourceDir, "blobs", filename)
		if _, err2 := os.Stat(alt); err2 == nil {
			srcPath = alt
		} else {
			return fmt.Errorf("blob not found: %s", filename)
		}
	}

	destPath := store.BlobPath(layer.Digest)

	if info, err := os.Stat(destPath); err == nil && info.Size() == layer.Size {
		fmt.Printf("  ↷ Skip  %s (%s)\n", filename, humanSize(layer.Size))
		return nil
	}

	if useSymlink {
		abs, _ := filepath.Abs(srcPath)
		fmt.Printf("  ⇢ Link  %s\n", filename)
		os.Remove(destPath)
		return os.Symlink(abs, destPath)
	}

	fmt.Printf("  → Copy  %s (%s)\n", filename, humanSize(layer.Size))
	return copyFile(srcPath, destPath)
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

func infoCmd() *cobra.Command {
	var registryBase string

	cmd := &cobra.Command{
		Use:   "info <model>",
		Short: "Show architecture, family, quantization, and parameters for a model",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ref, err := registry.ParseModelRef(args[0], registryBase)
			if err != nil {
				return err
			}

			fmt.Fprintf(os.Stderr, "→ Fetching manifest for %s\n", ref)
			manifest, rawManifest, err := registry.FetchManifest(ref)
			if err != nil {
				return err
			}

			fmt.Fprintf(os.Stderr, "→ Fetching config blob\n")
			cfg, err := registry.FetchConfig(ref, manifest)
			if err != nil {
				return fmt.Errorf("fetching config: %w", err)
			}

			_ = rawManifest

			fmt.Printf("\n╔══════════════════════════════════════╗\n")
			fmt.Printf("  Model Info: %s\n", ref)
			fmt.Printf("╚══════════════════════════════════════╝\n\n")

			printField := func(label, value string) {
				if value != "" {
					fmt.Printf("  %-22s %s\n", label+":", value)
				}
			}

			printField("Model Family", cfg.ModelFamily)
			if len(cfg.ModelFamilies) > 1 {
				printField("All Families", strings.Join(cfg.ModelFamilies, ", "))
			}
			printField("Architecture", cfg.Architecture)
			printField("Parameters", cfg.Parameters)
			printField("Format", cfg.ModelFormat)
			printField("File Type", cfg.FileType)
			printField("Quantization", cfg.QuantLevel)
			if cfg.Config.NumCtx > 0 {
				printField("Context Length", fmt.Sprintf("%d tokens", cfg.Config.NumCtx))
			}
			if len(cfg.Config.StopTokens) > 0 {
				printField("Stop Tokens", strings.Join(cfg.Config.StopTokens, " "))
			}

			fmt.Println()

			// Blob summary
			total := int64(0)
			for _, l := range manifest.AllLayers() {
				total += l.Size
			}
			fmt.Printf("  %-22s %d\n", "Blobs:", len(manifest.AllLayers()))
			fmt.Printf("  %-22s %s\n", "Total size:", humanSize(total))
			fmt.Println()

			fmt.Println("  Layers:")
			for _, l := range manifest.Layers {
				mt := strings.TrimPrefix(l.MediaType, "application/vnd.ollama.image.")
				fmt.Printf("    %-20s %s\n", mt, humanSize(l.Size))
			}

			return nil
		},
	}
	cmd.Flags().StringVar(&registryBase, "registry", "", "Override registry base URL")
	return cmd
}

// ---------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------

func diffCmd() *cobra.Command {
	var registryBase string

	cmd := &cobra.Command{
		Use:   "diff <model-a> <model-b>",
		Short: "Compare two model versions blob-by-blob",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			refA, err := registry.ParseModelRef(args[0], registryBase)
			if err != nil {
				return err
			}
			refB, err := registry.ParseModelRef(args[1], registryBase)
			if err != nil {
				return err
			}

			fmt.Fprintf(os.Stderr, "→ Fetching manifests...\n")
			manifestA, _, err := registry.FetchManifest(refA)
			if err != nil {
				return fmt.Errorf("fetching manifest for %s: %w", refA, err)
			}
			manifestB, _, err := registry.FetchManifest(refB)
			if err != nil {
				return fmt.Errorf("fetching manifest for %s: %w", refB, err)
			}

			digestsA := map[string]registry.Layer{}
			digestsB := map[string]registry.Layer{}

			for _, l := range manifestA.AllLayers() {
				digestsA[l.Digest] = l
			}
			for _, l := range manifestB.AllLayers() {
				digestsB[l.Digest] = l
			}

			fmt.Printf("\nDiff: %s  vs  %s\n\n", refA, refB)

			added := 0
			removed := 0
			shared := 0
			addedBytes := int64(0)
			removedBytes := int64(0)

			// In B but not A
			for d, l := range digestsB {
				if _, ok := digestsA[d]; !ok {
					mt := strings.TrimPrefix(l.MediaType, "application/vnd.ollama.image.")
					fmt.Printf("  + %-20s %-16s %s\n", mt, humanSize(l.Size), shortDigest(d))
					added++
					addedBytes += l.Size
				}
			}

			// In A but not B
			for d, l := range digestsA {
				if _, ok := digestsB[d]; !ok {
					mt := strings.TrimPrefix(l.MediaType, "application/vnd.ollama.image.")
					fmt.Printf("  - %-20s %-16s %s\n", mt, humanSize(l.Size), shortDigest(d))
					removed++
					removedBytes += l.Size
				}
			}

			// Shared
			for d := range digestsA {
				if _, ok := digestsB[d]; ok {
					shared++
				}
			}

			fmt.Printf("\n  Shared blobs    : %d (no download needed)\n", shared)
			fmt.Printf("  Added in B      : %d  (+%s)\n", added, humanSize(addedBytes))
			fmt.Printf("  Removed from A  : %d  (-%s)\n", removed, humanSize(removedBytes))

			if added == 0 && removed == 0 {
				fmt.Println("\n  ✅ Models are identical (same blobs)")
			} else {
				fmt.Printf("\n  Net change: %s\n", func() string {
					delta := addedBytes - removedBytes
					if delta >= 0 {
						return "+" + humanSize(delta)
					}
					return "-" + humanSize(-delta)
				}())
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&registryBase, "registry", "", "Override registry base URL")
	return cmd
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

func verifyCmd() *cobra.Command {
	var (
		ollamaDir string
		fix       bool
	)

	cmd := &cobra.Command{
		Use:   "verify [model]",
		Short: "SHA256-verify installed model blobs (all models if no arg given)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := ollama.NewStore(ollamaDir)
			if err != nil {
				return err
			}

			var refs []registry.ModelRef
			if len(args) == 1 {
				ref, err := registry.ParseModelRef(args[0], "")
				if err != nil {
					return err
				}
				refs = []registry.ModelRef{ref}
			} else {
				refs, err = store.InstalledModels()
				if err != nil {
					return fmt.Errorf("listing installed models: %w", err)
				}
			}

			if len(refs) == 0 {
				fmt.Println("No models installed.")
				return nil
			}

			totalOK, totalBad := 0, 0

			for _, ref := range refs {
				fmt.Printf("\nVerifying %s...\n", ref)
				manifest, _, err := store.ReadManifest(ref)
				if err != nil {
					fmt.Printf("  ✗ Cannot read manifest: %v\n", err)
					totalBad++
					continue
				}

				for _, layer := range manifest.AllLayers() {
					result := store.VerifyBlob(layer)
					switch result.Status {
					case "ok":
						fmt.Printf("  ✓ %s (%s)\n", result.Filename, humanSize(result.Size))
						totalOK++
					default:
						fmt.Printf("  ✗ %s — %s: %s\n", result.Filename, result.Status, result.Detail)
						totalBad++
						if fix {
							os.Remove(store.BlobPath(layer.Digest))
							fmt.Printf("    ↳ Removed (re-run 'ollama pull %s' to repair)\n", ref)
						}
					}
				}
			}

			fmt.Printf("\n── Summary ──\n")
			fmt.Printf("  OK      : %d\n", totalOK)
			fmt.Printf("  Corrupt : %d\n", totalBad)

			if totalBad > 0 {
				if fix {
					fmt.Println("\nCorrupt blobs removed. Run 'ollama pull <model>' to re-download them.")
				} else {
					fmt.Println("\nRe-run with --fix to remove corrupt blobs.")
				}
				return fmt.Errorf("%d corrupt blob(s) found", totalBad)
			}
			fmt.Println("  ✅ All blobs verified OK")
			return nil
		},
	}

	cmd.Flags().StringVar(&ollamaDir, "ollama-dir", "", "Override Ollama models directory")
	cmd.Flags().BoolVar(&fix, "fix", false, "Remove corrupt blobs so they can be re-pulled")
	return cmd
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

func searchCmd() *cobra.Command {
	var jsonOutput bool

	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Search the Ollama library for available models",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprintf(os.Stderr, "→ Searching Ollama library for %q...\n\n", args[0])

			results, err := registry.SearchModels(args[0])
			if err != nil {
				return err
			}

			if len(results) == 0 {
				fmt.Println("No results found.")
				return nil
			}

			if jsonOutput {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(results)
			}

			for _, r := range results {
				fmt.Printf("  %-35s  %s\n", r.Name, r.Description)
				if len(r.Tags) > 0 {
					shown := r.Tags
					if len(shown) > 6 {
						shown = shown[:6]
					}
					fmt.Printf("    Tags: %s\n", strings.Join(shown, ", "))
				}
				if r.Pulls > 0 {
					fmt.Printf("    Pulls: %s\n", formatPulls(r.Pulls))
				}
				fmt.Println()
			}

			return nil
		},
	}
	cmd.Flags().BoolVar(&jsonOutput, "json", false, "JSON output")
	return cmd
}

// ---------------------------------------------------------------------------
// pack
// ---------------------------------------------------------------------------

func packCmd() *cobra.Command {
	var (
		outFile      string
		ollamaDir    string
		compLevel    int
		registryBase string
	)

	cmd := &cobra.Command{
		Use:   "pack <model>",
		Short: "Bundle an installed model into a portable .tar.zst archive",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ref, err := registry.ParseModelRef(args[0], registryBase)
			if err != nil {
				return err
			}

			store, err := ollama.NewStore(ollamaDir)
			if err != nil {
				return err
			}

			manifest, rawManifest, err := store.ReadManifest(ref)
			if err != nil {
				return fmt.Errorf("model not installed locally: %w", err)
			}

			if outFile == "" {
				outFile = pack.ArchiveName(ref)
			}

			fmt.Printf("Packing %s → %s\n\n", ref, outFile)

			if err := pack.Pack(manifest, rawManifest, store.BlobsDir(), outFile, pack.PackOptions{
				CompressionLevel: compLevel,
			}); err != nil {
				return err
			}

			info, _ := os.Stat(outFile)
			fmt.Printf("\n✅ Archive created: %s (%s)\n", outFile, humanSize(info.Size()))
			return nil
		},
	}

	cmd.Flags().StringVarP(&outFile, "out", "o", "", "Output archive path")
	cmd.Flags().StringVar(&ollamaDir, "ollama-dir", "", "Override Ollama models directory")
	cmd.Flags().IntVar(&compLevel, "compression", 3, "zstd compression level (1=fast, 22=best)")
	cmd.Flags().StringVar(&registryBase, "registry", "", "Override registry base URL")
	return cmd
}

// ---------------------------------------------------------------------------
// unpack
// ---------------------------------------------------------------------------

func unpackCmd() *cobra.Command {
	var (
		ollamaDir string
		modelName string
		inspect   bool
	)

	cmd := &cobra.Command{
		Use:   "unpack <archive.tar.zst>",
		Short: "Extract and install a packed model archive",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			archivePath := args[0]

			if inspect {
				return pack.Inspect(archivePath)
			}

			if modelName == "" {
				return fmt.Errorf("--name is required (e.g. --name mymodel:local)")
			}

			tmpDir, err := os.MkdirTemp("", "ollama-dl-unpack-*")
			if err != nil {
				return err
			}
			defer os.RemoveAll(tmpDir)

			fmt.Printf("Extracting %s...\n\n", archivePath)
			rawManifest, err := pack.Unpack(archivePath, tmpDir)
			if err != nil {
				return err
			}

			// Write manifest.json to tmpDir root for installCmd reuse
			if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), rawManifest, 0o644); err != nil {
				return err
			}

			// Move blobs subdir up if present
			blobsSubdir := filepath.Join(tmpDir, "blobs")
			if _, err := os.Stat(blobsSubdir); os.IsNotExist(err) {
				blobsSubdir = tmpDir
			}

			store, err := ollama.NewStore(ollamaDir)
			if err != nil {
				return err
			}

			var manifest registry.Manifest
			if err := json.Unmarshal(rawManifest, &manifest); err != nil {
				return err
			}

			ref, err := registry.ParseModelRef(modelName, "")
			if err != nil {
				return err
			}

			if err := os.MkdirAll(store.BlobsDir(), 0o755); err != nil {
				return err
			}

			fmt.Printf("\nInstalling into Ollama store...\n")
			for _, layer := range manifest.AllLayers() {
				if err := installBlob(store, blobsSubdir, layer, false); err != nil {
					// Try parent tmpDir too
					if err2 := installBlob(store, tmpDir, layer, false); err2 != nil {
						return err
					}
				}
			}

			if err := store.WriteManifest(ref, rawManifest); err != nil {
				return err
			}

			fmt.Printf("\n✅ %s installed!\n", ref)
			return nil
		},
	}

	cmd.Flags().StringVar(&ollamaDir, "ollama-dir", "", "Override Ollama models directory")
	cmd.Flags().StringVar(&modelName, "name", "", "Model name to register (required unless --inspect)")
	cmd.Flags().BoolVar(&inspect, "inspect", false, "Print archive contents without extracting")
	return cmd
}

// ---------------------------------------------------------------------------
// batch
// ---------------------------------------------------------------------------

// BatchManifest is the schema for a YAML/JSON batch download file.
type BatchManifest struct {
	OutDir  string        `yaml:"out_dir"  json:"out_dir"`
	Models  []BatchModel  `yaml:"models"   json:"models"`
}

type BatchModel struct {
	Model       string `yaml:"model"        json:"model"`
	Name        string `yaml:"name"         json:"name"`          // install name override
	Registry    string `yaml:"registry"     json:"registry"`
	AutoInstall bool   `yaml:"auto_install" json:"auto_install"`
}

func batchCmd() *cobra.Command {
	var (
		ollamaDir   string
		concurrency int
		dryRun      bool
	)

	cmd := &cobra.Command{
		Use:   "batch <manifest.yaml>",
		Short: "Download (and optionally install) multiple models from a YAML/JSON file",
		Long: `Downloads multiple models defined in a batch manifest file.

Example manifest.yaml:
  out_dir: ./models
  models:
    - model: llama3:8b
      auto_install: true
    - model: mistral:7b-instruct-v0.2-q5_K_M
      name: mistral-q5:local
      auto_install: true
    - model: myorg/custom-model:v2
      registry: https://my-registry.example.com`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			data, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("reading batch file: %w", err)
			}

			var bm BatchManifest
			// Try YAML first, then JSON
			if err := yaml.Unmarshal(data, &bm); err != nil {
				if err2 := json.Unmarshal(data, &bm); err2 != nil {
					return fmt.Errorf("parsing batch file as YAML (%v) or JSON (%v)", err, err2)
				}
			}

			if len(bm.Models) == 0 {
				return fmt.Errorf("batch file contains no models")
			}

			if bm.OutDir == "" {
				bm.OutDir = "./ollama-downloads"
			}

			fmt.Printf("Batch: %d model(s) → %s\n\n", len(bm.Models), bm.OutDir)

			if dryRun {
				for _, m := range bm.Models {
					fmt.Printf("  would download: %s", m.Model)
					if m.AutoInstall {
						fmt.Printf(" (auto-install as %s)", coalesce(m.Name, m.Model))
					}
					fmt.Println()
				}
				return nil
			}

			// Track which blobs have already been downloaded (dedup across models)
			downloadedBlobs := map[string]string{} // digest → local path

			store, _ := ollama.NewStore(ollamaDir)

			for i, bmodel := range bm.Models {
				fmt.Printf("── Model %d/%d: %s ────────────────────────\n", i+1, len(bm.Models), bmodel.Model)

				ref, err := registry.ParseModelRef(bmodel.Model, bmodel.Registry)
				if err != nil {
					fmt.Printf("  ✗ %v\n\n", err)
					continue
				}

				modelDir := filepath.Join(bm.OutDir, sanitizeDirName(ref.Repo+"-"+ref.Tag))
				if err := os.MkdirAll(modelDir, 0o755); err != nil {
					fmt.Printf("  ✗ %v\n\n", err)
					continue
				}

				manifest, rawManifest, err := registry.FetchManifest(ref)
				if err != nil {
					fmt.Printf("  ✗ fetching manifest: %v\n\n", err)
					continue
				}

				if err := os.WriteFile(filepath.Join(modelDir, "manifest.json"), rawManifest, 0o644); err != nil {
					fmt.Printf("  ✗ saving manifest: %v\n\n", err)
					continue
				}

				// Deduplicate: if we already downloaded a blob, symlink it
				var jobs []download.Job
				for _, layer := range manifest.AllLayers() {
					dest := filepath.Join(modelDir, registry.DigestToFilename(layer.Digest))
					if existingPath, ok := downloadedBlobs[layer.Digest]; ok {
						// Already downloaded in a previous model — symlink
						if _, err := os.Stat(dest); os.IsNotExist(err) {
							abs, _ := filepath.Abs(existingPath)
							os.Symlink(abs, dest)
							fmt.Printf("  ⇢ Dedup  %s\n", registry.DigestToFilename(layer.Digest))
						}
						continue
					}
					if info, err := os.Stat(dest); err == nil && info.Size() == layer.Size {
						downloadedBlobs[layer.Digest] = dest
						continue
					}
					jobs = append(jobs, download.Job{
						URL:    ref.BlobURL(layer.Digest),
						Dest:   dest,
						Digest: layer.Digest,
						Size:   layer.Size,
						Label:  ref.Repo + " " + registry.DigestToFilename(layer.Digest)[:16],
					})
				}

				if len(jobs) > 0 {
					results := download.Run(context.Background(), jobs, download.Options{
						Concurrency: concurrency,
					})
					fmt.Println()
					for _, r := range results {
						if r.Err != nil {
							fmt.Printf("  ✗ %s: %v\n", r.Job.Label, r.Err)
						} else {
							downloadedBlobs[r.Job.Digest] = r.Job.Dest
						}
					}
				}

				if bmodel.AutoInstall && store != nil {
					installName := coalesce(bmodel.Name, bmodel.Model)
					installRef, err := registry.ParseModelRef(installName, "")
					if err == nil {
						_ = os.MkdirAll(store.BlobsDir(), 0o755)
						for _, layer := range manifest.AllLayers() {
							installBlob(store, modelDir, layer, false)
						}
						store.WriteManifest(installRef, rawManifest)
						fmt.Printf("  ✓ Installed as %s\n", installRef)
					}
				}
				fmt.Println()
			}

			fmt.Println("✅ Batch complete.")
			return nil
		},
	}

	cmd.Flags().StringVar(&ollamaDir, "ollama-dir", "", "Override Ollama models directory")
	cmd.Flags().IntVarP(&concurrency, "concurrency", "j", 3, "Parallel downloads per model")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "Print what would be downloaded without doing it")
	return cmd
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		os.Remove(dst)
		return err
	}
	return out.Sync()
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

func shortDigest(d string) string {
	parts := strings.SplitN(d, ":", 2)
	if len(parts) == 2 && len(parts[1]) >= 12 {
		return parts[0] + ":" + parts[1][:12] + "..."
	}
	return d
}

func formatPulls(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func sanitizeDirName(s string) string {
	r := strings.NewReplacer("/", "_", ":", "_", " ", "_")
	return r.Replace(s)
}

func coalesce(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
