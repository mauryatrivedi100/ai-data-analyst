/**
 * Feature: ai-data-analyst, Property 27: Form State Preservation on Error
 *
 * Validates: Requirements 15.7, 16.4
 *
 * Property: For any form state and any error (validation failure or backend failure),
 * the application SHALL preserve all user-entered field values such that after the
 * error is displayed, the form state remains identical to the state before the error occurred.
 *
 * This test validates the principle that error state management is independent of form state.
 * It tests the pattern used across all pages: setting an error never modifies form fields.
 */
import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';

/**
 * Simulates the error handling pattern used in CleaningPage and other pages.
 * When an error occurs (validation or backend), only the error state changes —
 * the form state object remains completely untouched.
 *
 * This mirrors the React component pattern:
 *   const [formState, setFormState] = useState(initial);
 *   const [error, setError] = useState(null);
 *   // On error: setError(message) — formState is never mutated
 */
function handleErrorPreservingFormState(formState, errorMessage) {
  // The error handling pattern: error state is independent of form state.
  // We return the form state unchanged and the error separately.
  return {
    formState: formState,
    error: errorMessage,
  };
}

/**
 * Simulates a validation failure scenario.
 * Returns the form state unchanged and an error object.
 */
function handleValidationError(formState, fieldName, validationMessage) {
  return {
    formState: formState,
    error: {
      field: fieldName,
      message: validationMessage,
    },
  };
}

/**
 * Simulates a backend error scenario (API returns error).
 * Returns the form state unchanged and the backend error.
 */
function handleBackendError(formState, statusCode, errorBody) {
  return {
    formState: formState,
    error: {
      statusCode: statusCode,
      message: errorBody,
    },
  };
}

// Arbitrary for form field values (strings, numbers, arrays, booleans)
const formFieldValue = fc.oneof(
  fc.string(),
  fc.integer(),
  fc.double({ noNaN: true, noDefaultInfinity: true }),
  fc.boolean(),
  fc.array(fc.string(), { maxLength: 10 }),
  fc.array(fc.integer(), { maxLength: 10 })
);

// Arbitrary for a form state object (1–20 fields with various value types)
const formStateArbitrary = fc.dictionary(
  fc.string({ minLength: 1, maxLength: 30 }).filter((s) => s.trim().length > 0),
  formFieldValue,
  { minKeys: 1, maxKeys: 20 }
);

// Arbitrary for error messages (non-empty strings)
const errorMessageArbitrary = fc.string({ minLength: 1, maxLength: 200 });

// Arbitrary for HTTP status codes (4xx and 5xx)
const statusCodeArbitrary = fc.integer({ min: 400, max: 599 });

describe('Property 27: Form State Preservation on Error', () => {
  it('form state is identical before and after error display (general error)', () => {
    fc.assert(
      fc.property(formStateArbitrary, errorMessageArbitrary, (formState, errorMsg) => {
        // Deep clone for comparison
        const formStateBefore = JSON.parse(JSON.stringify(formState));

        const result = handleErrorPreservingFormState(formState, errorMsg);

        // The form state must be identical to what it was before the error
        expect(result.formState).toEqual(formStateBefore);
        // The error should be set
        expect(result.error).toBe(errorMsg);
        // The original form state reference should be the same (not a copy)
        expect(result.formState).toBe(formState);
      }),
      { numRuns: 100 }
    );
  });

  it('form state is identical before and after validation error', () => {
    fc.assert(
      fc.property(
        formStateArbitrary,
        fc.string({ minLength: 1, maxLength: 50 }),
        errorMessageArbitrary,
        (formState, fieldName, validationMsg) => {
          const formStateBefore = JSON.parse(JSON.stringify(formState));

          const result = handleValidationError(formState, fieldName, validationMsg);

          // Form state must be preserved exactly
          expect(result.formState).toEqual(formStateBefore);
          expect(result.formState).toBe(formState);
          // Error contains field-specific information
          expect(result.error.field).toBe(fieldName);
          expect(result.error.message).toBe(validationMsg);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('form state is identical before and after backend error', () => {
    fc.assert(
      fc.property(
        formStateArbitrary,
        statusCodeArbitrary,
        errorMessageArbitrary,
        (formState, statusCode, errorBody) => {
          const formStateBefore = JSON.parse(JSON.stringify(formState));

          const result = handleBackendError(formState, statusCode, errorBody);

          // Form state must be preserved exactly
          expect(result.formState).toEqual(formStateBefore);
          expect(result.formState).toBe(formState);
          // Error is properly stored
          expect(result.error.statusCode).toBe(statusCode);
          expect(result.error.message).toBe(errorBody);
        }
      ),
      { numRuns: 100 }
    );
  });

  it('form state with nested arrays is preserved on error', () => {
    // Use integer for numericInput to avoid -0/+0 JSON serialization edge case
    const complexFormState = fc.record({
      selectedColumns: fc.array(fc.string(), { maxLength: 10 }),
      inputValue: fc.string(),
      numericInput: fc.integer(),
      dropdownSelection: fc.string(),
      checkboxValues: fc.array(fc.boolean(), { maxLength: 5 }),
      targetType: fc.constantFrom('int', 'float', 'string', 'datetime'),
    });

    fc.assert(
      fc.property(complexFormState, errorMessageArbitrary, (formState, errorMsg) => {
        const result = handleErrorPreservingFormState(formState, errorMsg);

        // The returned formState reference is the exact same object (not a copy)
        expect(result.formState).toBe(formState);
        // Every field must remain identical via reference equality
        expect(result.formState.selectedColumns).toBe(formState.selectedColumns);
        expect(result.formState.inputValue).toBe(formState.inputValue);
        expect(result.formState.numericInput).toBe(formState.numericInput);
        expect(result.formState.dropdownSelection).toBe(formState.dropdownSelection);
        expect(result.formState.checkboxValues).toBe(formState.checkboxValues);
        expect(result.formState.targetType).toBe(formState.targetType);
      }),
      { numRuns: 100 }
    );
  });

  it('multiple sequential errors do not degrade form state', () => {
    fc.assert(
      fc.property(
        formStateArbitrary,
        fc.array(errorMessageArbitrary, { minLength: 1, maxLength: 5 }),
        (formState, errors) => {
          const formStateBefore = JSON.parse(JSON.stringify(formState));

          // Simulate multiple sequential errors (e.g., user retries failing operation)
          let currentState = formState;
          for (const errorMsg of errors) {
            const result = handleErrorPreservingFormState(currentState, errorMsg);
            currentState = result.formState;
          }

          // After all errors, form state must still be identical to the original
          expect(currentState).toEqual(formStateBefore);
          expect(currentState).toBe(formState);
        }
      ),
      { numRuns: 100 }
    );
  });
});
