// Package registry handles communication with the Ollama OCI registry.
package registry

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	DefaultBase = "https://registry.ollama.ai"
	SearchBase  = "https://ollama.com"
)

// Manifest is an OCI image manifest as returned by registry.ollama.ai.
type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`
}

// AllLayers returns Config + Layers as a single slice.
func (m *Manifest) AllLayers() []Layer {
	return append([]Layer{m.Config}, m.Layers...)
}

// Layer is one entry in a manifest (config or data blob).
type Layer struct {
	MediaType string `json:"mediaType"`
	Size      int64  `json:"size"`
	Digest    string `json:"digest"`
}

// ModelRef is a parsed model name.
type ModelRef struct {
	Namespace string
	Repo      string
	Tag       string
	Base      string // registry base URL
}

// String returns "namespace/repo:tag".
func (r ModelRef) String() string {
	return fmt.Sprintf("%s/%s:%s", r.Namespace, r.Repo, r.Tag)
}

// BlobURL returns the direct download URL for a digest.
func (r ModelRef) BlobURL(digest string) string {
	return fmt.Sprintf("%s/v2/%s/%s/blobs/%s", r.Base, r.Namespace, r.Repo, digest)
}

// ManifestURL returns the manifest URL for the ref's tag.
func (r ModelRef) ManifestURL() string {
	return fmt.Sprintf("%s/v2/%s/%s/manifests/%s", r.Base, r.Namespace, r.Repo, r.Tag)
}

// ParseModelRef parses a model name string into a ModelRef.
// Supported formats: repo, repo:tag, namespace/repo, namespace/repo:tag
func ParseModelRef(model string, base string) (ModelRef, error) {
	if base == "" {
		base = DefaultBase
	}
	ref := ModelRef{Namespace: "library", Tag: "latest", Base: base}

	parts := strings.SplitN(model, ":", 2)
	namepart := parts[0]
	if len(parts) == 2 {
		ref.Tag = parts[1]
	}

	slashParts := strings.SplitN(namepart, "/", 2)
	if len(slashParts) == 2 {
		ref.Namespace = slashParts[0]
		ref.Repo = slashParts[1]
	} else {
		ref.Repo = slashParts[0]
	}

	if ref.Repo == "" {
		return ModelRef{}, fmt.Errorf("invalid model name %q: repo cannot be empty", model)
	}
	return ref, nil
}

// FetchManifest retrieves and parses the OCI manifest for a model ref.
// Returns the parsed manifest and the raw JSON bytes (pretty-printed).
func FetchManifest(ref ModelRef) (*Manifest, []byte, error) {
	url := ref.ManifestURL()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("registry returned HTTP %d\nBody: %s", resp.StatusCode, string(body))
	}

	var m Manifest
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, nil, fmt.Errorf("parsing manifest JSON: %w", err)
	}

	pretty, _ := json.MarshalIndent(m, "", "  ")
	return &m, pretty, nil
}

// DigestToFilename converts "sha256:abc123" → "sha256-abc123"
func DigestToFilename(digest string) string {
	return strings.ReplaceAll(digest, ":", "-")
}

// ModelConfig is the JSON structure stored in the config blob.
type ModelConfig struct {
	ModelFormat    string         `json:"model_format"`
	ModelFamily    string         `json:"model_family"`
	ModelFamilies  []string       `json:"model_families"`
	ModelType      string         `json:"model_type"`
	FileType       string         `json:"file_type"`
	Architecture   string         `json:"architecture"`
	Parameters     string         `json:"parameters"`
	QuantLevel     string         `json:"quantization_level"`
	RootFS         RootFS         `json:"rootfs"`
	Config         InnerConfig    `json:"config"`
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

type InnerConfig struct {
	StopTokens []string `json:"stop"`
	NumCtx     int      `json:"num_ctx"`
}

// FetchConfig downloads and parses the config blob for a manifest.
func FetchConfig(ref ModelRef, manifest *Manifest) (*ModelConfig, error) {
	url := ref.BlobURL(manifest.Config.Digest)
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("fetching config blob: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("config blob returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var cfg ModelConfig
	if err := json.Unmarshal(body, &cfg); err != nil {
		return nil, fmt.Errorf("parsing config JSON: %w", err)
	}
	return &cfg, nil
}

// SearchResult is one item from the Ollama model search API.
type SearchResult struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Tags        []string `json:"tags"`
	Pulls       int      `json:"pulls"`
	Updated     string   `json:"updated"`
}

// SearchModels queries the Ollama library search endpoint.
func SearchModels(query string) ([]SearchResult, error) {
	url := fmt.Sprintf("%s/api/models?q=%s", SearchBase, query)
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	// The API returns {"models": [...]}
	var wrapper struct {
		Models []SearchResult `json:"models"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		// Try as bare array
		var results []SearchResult
		if err2 := json.Unmarshal(body, &results); err2 != nil {
			return nil, fmt.Errorf("parsing search results: %w (raw: %s)", err, string(body[:min(200, len(body))]))
		}
		return results, nil
	}
	return wrapper.Models, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
