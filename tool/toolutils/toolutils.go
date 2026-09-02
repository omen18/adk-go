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

// Package toolutils provides public helpers for packing tool declarations
// into a model.LLMRequest. It allows external code to consolidate tool
// function declarations the same way the built-in ADK tools do, without
// re-implementing the logic.
package toolutils

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// Tool is implemented by any tool that can be packed into a
// model.LLMRequest via PackTool.
type Tool interface {
	Name() string
	Declaration() *genai.FunctionDeclaration
}

// PackTool ensures that in case there is a usage of multiple function tools,
// all of them are consolidated into one genai tool that has all the function declarations
// provided by the tools. So, if there is already a tool with a function declaration,
// it appends another to it; otherwise, it creates a new genai tool.
func PackTool(req *model.LLMRequest, t Tool) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := t.Name()

	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if decl := t.Declaration(); decl == nil {
		return nil
	}
	// Find an existing genai.Tool with FunctionDeclarations
	var funcTool *genai.Tool
	for _, tool := range req.Config.Tools {
		if tool != nil && tool.FunctionDeclarations != nil {
			funcTool = tool
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{t.Declaration()},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, t.Declaration())
	}
	return nil
}

// GetRequiredStringParam extracts a required string parameter from a tool argument map.
func GetRequiredStringParam(m map[string]any, paramName string) (string, error) {
	if m == nil {
		return "", fmt.Errorf("arguments map is nil")
	}
	val, ok := m[paramName]
	if !ok {
		return "", fmt.Errorf("missing required parameter %q", paramName)
	}
	strVal, ok := val.(string)
	if !ok {
		return "", fmt.Errorf("parameter %q must be a string, got %T", paramName, val)
	}
	if strVal == "" {
		return "", fmt.Errorf("parameter %q cannot be empty", paramName)
	}
	return strVal, nil
}

// GetOptionalStringParam extracts an optional string parameter from a tool argument map.
// If the key is missing or not a string or empty, it returns defaultVal.
func GetOptionalStringParam(m map[string]any, paramName string, defaultVal string) string {
	if m == nil {
		return defaultVal
	}
	val, ok := m[paramName]
	if !ok {
		return defaultVal
	}
	strVal, ok := val.(string)
	if !ok || strVal == "" {
		return defaultVal
	}
	return strVal
}

// GetRequiredIntParam extracts a required integer parameter from a tool argument map.
// Supports both int and float64 (from JSON unmarshaling).
func GetRequiredIntParam(m map[string]any, paramName string) (int, error) {
	if m == nil {
		return 0, fmt.Errorf("arguments map is nil")
	}
	val, ok := m[paramName]
	if !ok {
		return 0, fmt.Errorf("missing required parameter %q", paramName)
	}
	switch v := val.(type) {
	case int:
		return v, nil
	case int32:
		return int(v), nil
	case int64:
		return int(v), nil
	case float64:
		return int(v), nil
	default:
		return 0, fmt.Errorf("parameter %q must be an integer, got %T", paramName, val)
	}
}

// GetOptionalIntParam extracts an optional integer parameter from a tool argument map.
func GetOptionalIntParam(m map[string]any, paramName string, defaultVal int) (int, error) {
	if m == nil {
		return defaultVal, nil
	}
	val, ok := m[paramName]
	if !ok {
		return defaultVal, nil
	}
	switch v := val.(type) {
	case int:
		return v, nil
	case int32:
		return int(v), nil
	case int64:
		return int(v), nil
	case float64:
		return int(v), nil
	default:
		return defaultVal, fmt.Errorf("parameter %q must be an integer, got %T", paramName, val)
	}
}

// GetRequiredBoolParam extracts a required boolean parameter from a tool argument map.
func GetRequiredBoolParam(m map[string]any, paramName string) (bool, error) {
	if m == nil {
		return false, fmt.Errorf("arguments map is nil")
	}
	val, ok := m[paramName]
	if !ok {
		return false, fmt.Errorf("missing required parameter %q", paramName)
	}
	boolVal, ok := val.(bool)
	if !ok {
		return false, fmt.Errorf("parameter %q must be a boolean, got %T", paramName, val)
	}
	return boolVal, nil
}

// GetOptionalBoolParam extracts an optional boolean parameter from a tool argument map.
func GetOptionalBoolParam(m map[string]any, paramName string, defaultVal bool) (bool, error) {
	if m == nil {
		return defaultVal, nil
	}
	val, ok := m[paramName]
	if !ok {
		return defaultVal, nil
	}
	boolVal, ok := val.(bool)
	if !ok {
		return defaultVal, fmt.Errorf("parameter %q must be a boolean, got %T", paramName, val)
	}
	return boolVal, nil
}

// GetRequiredFloatParam extracts a required float64 parameter from a tool argument map.
func GetRequiredFloatParam(m map[string]any, paramName string) (float64, error) {
	if m == nil {
		return 0, fmt.Errorf("arguments map is nil")
	}
	val, ok := m[paramName]
	if !ok {
		return 0, fmt.Errorf("missing required parameter %q", paramName)
	}
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	default:
		return 0, fmt.Errorf("parameter %q must be a float, got %T", paramName, val)
	}
}

// GetOptionalFloatParam extracts an optional float64 parameter from a tool argument map.
func GetOptionalFloatParam(m map[string]any, paramName string, defaultVal float64) (float64, error) {
	if m == nil {
		return defaultVal, nil
	}
	val, ok := m[paramName]
	if !ok {
		return defaultVal, nil
	}
	switch v := val.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	default:
		return defaultVal, fmt.Errorf("parameter %q must be a float, got %T", paramName, val)
	}
}

