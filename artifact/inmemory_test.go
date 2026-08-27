// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package artifact_test

import (
	"fmt"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
)

func TestInMemoryArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return artifact.InMemoryService(), nil
	}
	tests.TestArtifactService(t, "InMemory", factory)
}

func TestInMemoryArtifactService_Concurrent(t *testing.T) {
	ctx := t.Context()
	s := artifact.InMemoryService()

	const workers = 8
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(2)
		// Writer goroutine
		go func(w int) {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				fileName := fmt.Sprintf("file_%d_%d.txt", w, j)
				_, err := s.Save(ctx, &artifact.SaveRequest{
					AppName:   "app1",
					UserID:    "user1",
					SessionID: "sess1",
					FileName:  fileName,
					Part:      genai.NewPartFromText("data"),
				})
				if err != nil {
					t.Errorf("Save error: %v", err)
					return
				}
			}
		}(i)

		// Reader / Lister goroutine
		go func(w int) {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				_, _ = s.List(ctx, &artifact.ListRequest{
					AppName:   "app1",
					UserID:    "user1",
					SessionID: "sess1",
				})
			}
		}(i)
	}

	wg.Wait()
}
