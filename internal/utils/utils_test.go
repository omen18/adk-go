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

package utils

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestPopulateAndRemoveClientFunctionCallID(t *testing.T) {
	ctx := context.Background()
	content := &genai.Content{
		Parts: []*genai.Part{
			{
				FunctionCall: &genai.FunctionCall{
					Name: "test_func",
				},
			},
		},
	}

	PopulateClientFunctionCallID(ctx, content)
	calls := FunctionCalls(content)
	if len(calls) != 1 {
		t.Fatalf("expected 1 function call, got %d", len(calls))
	}
	if calls[0].ID == "" {
		t.Errorf("expected non-empty function call ID after population")
	}
	if !strings.HasPrefix(calls[0].ID, afFunctionCallIDPrefix) {
		t.Errorf("expected ID prefix %q, got %q", afFunctionCallIDPrefix, calls[0].ID)
	}

	RemoveClientFunctionCallID(content)
	if calls[0].ID != "" {
		t.Errorf("expected empty function call ID after removal, got %q", calls[0].ID)
	}
}

func TestContentAndPartExtractors(t *testing.T) {
	if Content(nil) != nil {
		t.Errorf("Content(nil) should be nil")
	}

	ev := &session.Event{
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("hello world"),
					{
						FunctionCall: &genai.FunctionCall{Name: "search"},
					},
					{
						FunctionResponse: &genai.FunctionResponse{Name: "search", Response: map[string]any{"res": "ok"}},
					},
				},
			},
		},
	}

	c := Content(ev)
	if c == nil {
		t.Fatalf("Content(ev) should not be nil")
	}

	texts := TextParts(c)
	if len(texts) != 1 || texts[0] != "hello world" {
		t.Errorf("unexpected text parts: %v", texts)
	}

	fnCalls := FunctionCalls(c)
	if len(fnCalls) != 1 || fnCalls[0].Name != "search" {
		t.Errorf("unexpected function calls: %v", fnCalls)
	}

	fnResps := FunctionResponses(c)
	if len(fnResps) != 1 || fnResps[0].Name != "search" {
		t.Errorf("unexpected function responses: %v", fnResps)
	}
}

func TestFunctionDecls(t *testing.T) {
	if decls := FunctionDecls(nil); len(decls) != 0 {
		t.Errorf("FunctionDecls(nil) should return empty slice")
	}

	cfgWithNilTool := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			nil,
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "f1"},
				},
			},
		},
	}
	declsNil := FunctionDecls(cfgWithNilTool)
	if len(declsNil) != 1 || declsNil[0].Name != "f1" {
		t.Errorf("unexpected function decls with nil tool: %v", declsNil)
	}

	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{Name: "f1"},
				},
			},
		},
	}

	decls := FunctionDecls(cfg)
	if len(decls) != 1 || decls[0].Name != "f1" {
		t.Errorf("unexpected function decls: %v", decls)
	}
}

func TestAppendInstructions(t *testing.T) {
	// Should safely handle nil request
	AppendInstructions(nil, "some instructions")

	req := &model.LLMRequest{}
	AppendInstructions(req, "Instruction 1", "Instruction 2")

	if req.Config == nil || req.Config.SystemInstruction == nil {
		t.Fatalf("expected SystemInstruction to be set")
	}

	texts := TextParts(req.Config.SystemInstruction)
	if len(texts) == 0 {
		t.Fatalf("expected non-empty text in SystemInstruction")
	}

	expected := "Instruction 1\n\nInstruction 2"
	if !strings.Contains(texts[0], expected) {
		t.Errorf("expected instruction content %q in %q", expected, texts[0])
	}
}

func TestMust(t *testing.T) {
	val := Must("hello", nil)
	if val != "hello" {
		t.Errorf("Must return value mismatch: got %v", val)
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("Must should have panicked on error")
		}
	}()

	_ = Must(123, errors.New("test error"))
}
