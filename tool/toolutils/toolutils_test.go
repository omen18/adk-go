// Copyright 2026 Google LLC
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

package toolutils_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// fakeTool is a minimal implementation of toolutils.Tool for tests.
type fakeTool struct {
	name string
	decl *genai.FunctionDeclaration
}

func (f fakeTool) Name() string                            { return f.name }
func (f fakeTool) Declaration() *genai.FunctionDeclaration { return f.decl }

var _ toolutils.Tool = fakeTool{}

func TestPackTool_SingleTool(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "first", decl: &genai.FunctionDeclaration{Name: "first"}}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("PackTool() returned error: %v", err)
	}

	if _, ok := req.Tools["first"]; !ok {
		t.Fatalf("req.Tools missing %q; got %v", "first", req.Tools)
	}
	if got := len(req.Config.Tools); got != 1 {
		t.Fatalf("len(req.Config.Tools) = %d, want 1", got)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 1 {
		t.Fatalf("len(FunctionDeclarations) = %d, want 1", got)
	}
}

func TestPackTool_SecondToolSharesSameGenaiTool(t *testing.T) {
	req := &model.LLMRequest{}
	first := fakeTool{name: "first", decl: &genai.FunctionDeclaration{Name: "first"}}
	second := fakeTool{name: "second", decl: &genai.FunctionDeclaration{Name: "second"}}

	if err := toolutils.PackTool(req, first); err != nil {
		t.Fatalf("PackTool(first) returned error: %v", err)
	}
	if err := toolutils.PackTool(req, second); err != nil {
		t.Fatalf("PackTool(second) returned error: %v", err)
	}

	if _, ok := req.Tools["second"]; !ok {
		t.Fatalf("req.Tools missing %q; got %v", "second", req.Tools)
	}
	// Both declarations must live on a single genai.Tool, not two.
	if got := len(req.Config.Tools); got != 1 {
		t.Fatalf("len(req.Config.Tools) = %d, want 1 (second tool should append, not create)", got)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 2 {
		t.Fatalf("len(FunctionDeclarations) = %d, want 2", got)
	}
}

func TestPackTool_DuplicateName(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "dup", decl: &genai.FunctionDeclaration{Name: "dup"}}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("first PackTool() returned error: %v", err)
	}
	if err := toolutils.PackTool(req, tool); err == nil {
		t.Fatalf("second PackTool() with duplicate name = nil error, want error")
	}
}

func TestPackTool_NilDeclaration(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "nodecl", decl: nil}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("PackTool() returned error: %v", err)
	}

	if _, ok := req.Tools["nodecl"]; !ok {
		t.Fatalf("req.Tools missing %q; nil-declaration tool should still register", "nodecl")
	}
	if got := len(req.Config.Tools); got != 0 {
		t.Fatalf("len(req.Config.Tools) = %d, want 0 (nil declaration adds nothing)", got)
	}
}

func TestGetRequiredStringParam(t *testing.T) {
	m := map[string]any{
		"valid": "hello",
		"empty": "",
		"num":   123,
	}

	val, err := toolutils.GetRequiredStringParam(m, "valid")
	if err != nil || val != "hello" {
		t.Errorf("expected 'hello', got %q, err=%v", val, err)
	}

	if _, err := toolutils.GetRequiredStringParam(m, "empty"); err == nil {
		t.Errorf("expected error for empty string")
	}

	if _, err := toolutils.GetRequiredStringParam(m, "missing"); err == nil {
		t.Errorf("expected error for missing parameter")
	}

	if _, err := toolutils.GetRequiredStringParam(m, "num"); err == nil {
		t.Errorf("expected error for non-string parameter")
	}

	if _, err := toolutils.GetRequiredStringParam(nil, "valid"); err == nil {
		t.Errorf("expected error for nil map")
	}
}

func TestGetOptionalStringParam(t *testing.T) {
	m := map[string]any{
		"valid": "hello",
		"empty": "",
		"num":   123,
	}

	if got := toolutils.GetOptionalStringParam(m, "valid", "default"); got != "hello" {
		t.Errorf("expected 'hello', got %q", got)
	}
	if got := toolutils.GetOptionalStringParam(m, "empty", "default"); got != "default" {
		t.Errorf("expected 'default', got %q", got)
	}
	if got := toolutils.GetOptionalStringParam(m, "missing", "default"); got != "default" {
		t.Errorf("expected 'default', got %q", got)
	}
	if got := toolutils.GetOptionalStringParam(m, "num", "default"); got != "default" {
		t.Errorf("expected 'default', got %q", got)
	}
	if got := toolutils.GetOptionalStringParam(nil, "key", "default"); got != "default" {
		t.Errorf("expected 'default', got %q", got)
	}
}

func TestGetRequiredIntParam(t *testing.T) {
	m := map[string]any{
		"int":     42,
		"int32":   int32(42),
		"int64":   int64(42),
		"float64": float64(42),
		"str":     "42",
	}

	for _, key := range []string{"int", "int32", "int64", "float64"} {
		val, err := toolutils.GetRequiredIntParam(m, key)
		if err != nil || val != 42 {
			t.Errorf("key %q: expected 42, got %d, err=%v", key, val, err)
		}
	}

	if _, err := toolutils.GetRequiredIntParam(m, "str"); err == nil {
		t.Errorf("expected error for non-int parameter")
	}
	if _, err := toolutils.GetRequiredIntParam(m, "missing"); err == nil {
		t.Errorf("expected error for missing parameter")
	}
	if _, err := toolutils.GetRequiredIntParam(nil, "int"); err == nil {
		t.Errorf("expected error for nil map")
	}
}

func TestGetOptionalIntParam(t *testing.T) {
	m := map[string]any{
		"int":   100,
		"float": float64(200),
		"bad":   "abc",
	}

	val, err := toolutils.GetOptionalIntParam(m, "int", 50)
	if err != nil || val != 100 {
		t.Errorf("expected 100, got %d, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalIntParam(m, "float", 50)
	if err != nil || val != 200 {
		t.Errorf("expected 200, got %d, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalIntParam(m, "missing", 50)
	if err != nil || val != 50 {
		t.Errorf("expected 50, got %d, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalIntParam(nil, "int", 50)
	if err != nil || val != 50 {
		t.Errorf("expected 50, got %d, err=%v", val, err)
	}
	if _, err := toolutils.GetOptionalIntParam(m, "bad", 50); err == nil {
		t.Errorf("expected error for bad type")
	}
}

func TestGetRequiredBoolParam(t *testing.T) {
	m := map[string]any{
		"trueVal":  true,
		"falseVal": false,
		"str":      "true",
	}

	val, err := toolutils.GetRequiredBoolParam(m, "trueVal")
	if err != nil || val != true {
		t.Errorf("expected true, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetRequiredBoolParam(m, "falseVal")
	if err != nil || val != false {
		t.Errorf("expected false, got %v, err=%v", val, err)
	}
	if _, err := toolutils.GetRequiredBoolParam(m, "str"); err == nil {
		t.Errorf("expected error for non-bool parameter")
	}
	if _, err := toolutils.GetRequiredBoolParam(m, "missing"); err == nil {
		t.Errorf("expected error for missing parameter")
	}
	if _, err := toolutils.GetRequiredBoolParam(nil, "key"); err == nil {
		t.Errorf("expected error for nil map")
	}
}

func TestGetOptionalBoolParam(t *testing.T) {
	m := map[string]any{
		"flag": true,
		"bad":  123,
	}

	val, err := toolutils.GetOptionalBoolParam(m, "flag", false)
	if err != nil || val != true {
		t.Errorf("expected true, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalBoolParam(m, "missing", true)
	if err != nil || val != true {
		t.Errorf("expected true, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalBoolParam(nil, "key", false)
	if err != nil || val != false {
		t.Errorf("expected false, got %v, err=%v", val, err)
	}
	if _, err := toolutils.GetOptionalBoolParam(m, "bad", false); err == nil {
		t.Errorf("expected error for bad type")
	}
}

func TestGetRequiredFloatParam(t *testing.T) {
	m := map[string]any{
		"f64": 3.14,
		"f32": float32(2.5),
		"int": 10,
		"str": "not-float",
	}

	val, err := toolutils.GetRequiredFloatParam(m, "f64")
	if err != nil || val != 3.14 {
		t.Errorf("expected 3.14, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetRequiredFloatParam(m, "f32")
	if err != nil || val != 2.5 {
		t.Errorf("expected 2.5, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetRequiredFloatParam(m, "int")
	if err != nil || val != 10.0 {
		t.Errorf("expected 10.0, got %v, err=%v", val, err)
	}
	if _, err := toolutils.GetRequiredFloatParam(m, "str"); err == nil {
		t.Errorf("expected error for non-float")
	}
	if _, err := toolutils.GetRequiredFloatParam(m, "missing"); err == nil {
		t.Errorf("expected error for missing parameter")
	}
	if _, err := toolutils.GetRequiredFloatParam(nil, "f64"); err == nil {
		t.Errorf("expected error for nil map")
	}
}

func TestGetOptionalFloatParam(t *testing.T) {
	m := map[string]any{
		"f64": 1.25,
		"bad": "invalid",
	}

	val, err := toolutils.GetOptionalFloatParam(m, "f64", 0.5)
	if err != nil || val != 1.25 {
		t.Errorf("expected 1.25, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalFloatParam(m, "missing", 0.5)
	if err != nil || val != 0.5 {
		t.Errorf("expected 0.5, got %v, err=%v", val, err)
	}
	val, err = toolutils.GetOptionalFloatParam(nil, "key", 0.5)
	if err != nil || val != 0.5 {
		t.Errorf("expected 0.5, got %v, err=%v", val, err)
	}
	if _, err := toolutils.GetOptionalFloatParam(m, "bad", 0.5); err == nil {
		t.Errorf("expected error for bad type")
	}
}

