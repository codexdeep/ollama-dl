// Package download provides a resumable, parallel blob downloader with progress bars.
package download

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"

	"github.com/user/ollama-dl/internal/registry"
)

// Job describes a single blob to download.
type Job struct {
	URL      string
	Dest     string // final destination path
	Digest   string
	Size     int64
	Label    string
}

// Result is the outcome of one download job.
type Result struct {
	Job Job
	Err error
}

// Options configures the downloader.
type Options struct {
	Concurrency int
	RetryMax    int
}

var DefaultOptions = Options{Concurrency: 3, RetryMax: 3}

// Run downloads all jobs concurrently, showing a live progress bar per blob.
// It returns a slice of results (one per job).
func Run(ctx context.Context, jobs []Job, opts Options) []Result {
	if opts.Concurrency <= 0 {
		opts.Concurrency = DefaultOptions.Concurrency
	}
	if opts.RetryMax <= 0 {
		opts.RetryMax = DefaultOptions.RetryMax
	}

	p := mpb.NewWithContext(ctx,
		mpb.WithWidth(60),
		mpb.WithRefreshRate(120*time.Millisecond),
	)

	sem := make(chan struct{}, opts.Concurrency)
	results := make([]Result, len(jobs))
	var wg sync.WaitGroup

	for i, job := range jobs {
		wg.Add(1)
		go func(idx int, j Job) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			bar := p.AddBar(j.Size,
				mpb.PrependDecorators(
					decor.Name(truncate(j.Label, 30), decor.WC{W: 32, C: decor.DidentRight}),
					decor.CountersKibiByte("% .2f / % .2f"),
				),
				mpb.AppendDecorators(
					decor.EwmaETA(decor.ET_STYLE_GO, 30),
					decor.Name(" "),
					decor.EwmaSpeed(decor.SizeB1024(0), "% .2f", 30),
				),
			)

			err := downloadWithRetry(ctx, j, bar, opts.RetryMax)
			if err != nil {
				bar.Abort(false)
			}
			results[idx] = Result{Job: j, Err: err}
		}(i, job)
	}

	wg.Wait()
	p.Wait()
	return results
}

// BuildJobs constructs download jobs for all blobs in a manifest.
// Blobs already present on disk with the correct size are skipped.
func BuildJobs(ref registry.ModelRef, manifest *registry.Manifest, destDir string) []Job {
	var jobs []Job
	for _, layer := range manifest.AllLayers() {
		filename := registry.DigestToFilename(layer.Digest)
		dest := filepath.Join(destDir, filename)

		// Skip if already fully downloaded
		if info, err := os.Stat(dest); err == nil && info.Size() == layer.Size {
			fmt.Printf("  ↷ Already downloaded: %s\n", filename)
			continue
		}

		label := layerLabel(layer.MediaType)
		jobs = append(jobs, Job{
			URL:    ref.BlobURL(layer.Digest),
			Dest:   dest,
			Digest: layer.Digest,
			Size:   layer.Size,
			Label:  label,
		})
	}
	return jobs
}

func downloadWithRetry(ctx context.Context, job Job, bar *mpb.Bar, maxRetries int) error {
	tmpPath := job.Dest + ".part"

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			fmt.Fprintf(os.Stderr, "  ↺ Retry %d/%d for %s\n", attempt, maxRetries, job.Label)
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}

		err := downloadOnce(ctx, job, tmpPath, bar)
		if err == nil {
			return os.Rename(tmpPath, job.Dest)
		}

		if attempt == maxRetries {
			os.Remove(tmpPath)
			return fmt.Errorf("after %d retries: %w", maxRetries, err)
		}
	}
	return nil
}

func downloadOnce(ctx context.Context, job Job, tmpPath string, bar *mpb.Bar) error {
	// Resume support: check partial file
	startByte := int64(0)
	if info, err := os.Stat(tmpPath); err == nil {
		startByte = info.Size()
		bar.SetCurrent(startByte)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", job.URL, nil)
	if err != nil {
		return err
	}
	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("server returned HTTP %d", resp.StatusCode)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if startByte > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		startByte = 0
		bar.SetCurrent(0)
	}

	f, err := os.OpenFile(tmpPath, flags, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	reader := bar.ProxyReader(resp.Body)
	defer reader.Close()

	_, err = io.Copy(f, reader)
	return err
}

func layerLabel(mediaType string) string {
	switch mediaType {
	case "application/vnd.ollama.image.model":
		return "model weights"
	case "application/vnd.ollama.image.template":
		return "template"
	case "application/vnd.ollama.image.params":
		return "parameters"
	case "application/vnd.ollama.image.system":
		return "system prompt"
	case "application/vnd.ollama.image.license":
		return "license"
	default:
		if mediaType != "" {
			return mediaType
		}
		return "config"
	}
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-1] + "…"
}
