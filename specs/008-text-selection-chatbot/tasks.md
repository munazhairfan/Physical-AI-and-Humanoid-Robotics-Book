# Tasks: Text Selection Chatbot Feature

**Feature**: Text Selection Chatbot Feature
**Branch**: `008-text-selection-chatbot`
**Date**: 2025-12-13
**Spec**: `./specs/008-text-selection-chatbot/spec.md`
**Plan**: `./specs/008-text-selection-chatbot/plan.md`

## Dependencies

- User Story 1 (P1) must be completed before User Story 2 (P2)
- User Story 2 (P2) must be completed before User Story 3 (P3)
- User Stories 1 and 2 can be developed in parallel with proper coordination

## Parallel Execution Examples

### Per Story:
- **P1**: UI implementation [T001-T004] can run in parallel with API integration [T005-T007]
- **P2**: Mobile detection [T008-T010] can run in parallel with button positioning [T011-T012]
- **P3**: Backend endpoint [T013-T014] can run in parallel with frontend integration [T015-T017]

## Implementation Strategy

**MVP Scope**: Complete User Story 1 (Desktop Text Selection) with minimal viable functionality including basic button appearance and simple integration with existing chatbot.

**Incremental Delivery**:
1. MVP: Desktop text selection with button appearance
2. Enhancement: Mobile long-press functionality
3. Integration: Full backend integration with selected text processing

---

## Phase 1: Setup

### Goal
Initialize project structure and verify existing dependencies for the text selection chatbot feature.

- [X] T001 Create feature branch 008-text-selection-chatbot
- [X] T002 Verify existing Docusaurus project structure in frontend/rag-chatbot-frontend/
- [X] T003 Verify existing backend structure with /selected_text endpoint in backend/app/main.py
- [X] T004 Confirm GEMINI_API_KEY is properly configured in environment

## Phase 2: Foundational

### Goal
Implement foundational components that are prerequisites for all user stories.

- [X] T005 Create JavaScript file for text selection functionality at frontend/rag-chatbot-frontend/static/js/selection-chatbot.js
- [X] T006 Configure Docusaurus to include selection-chatbot.js in all pages via docusaurus.config.ts
- [X] T007 Implement basic text selection detection functions in selection-chatbot.js
- [X] T008 [P] Implement button creation and styling functions in selection-chatbot.js
- [X] T009 [P] Set up event listeners for mouse and touch events in selection-chatbot.js

## Phase 3: [US1] Desktop Text Selection

### Goal
Implement desktop text selection functionality with floating "Ask Chatbot" button.

**Independent Test Criteria**: When user selects text on desktop, a "Ask Chatbot" button appears near the selection within 100ms.

- [X] T010 [US1] Implement desktop text selection detection using mouseup and selectionchange events in selection-chatbot.js
- [X] T011 [US1] Calculate selection position using getBoundingClientRect() in selection-chatbot.js
- [X] T012 [US1] Position floating button near selection with proper z-index in selection-chatbot.js
- [X] T013 [US1] Style the "Ask Chatbot" button with appropriate CSS in selection-chatbot.js
- [X] T014 [US1] Add smooth animations for button appearance/disappearance in selection-chatbot.js
- [X] T015 [US1] Implement button click handler to send selected text to chatbot in selection-chatbot.js
- [X] T016 [US1] Set selection threshold to minimum 10 characters in selection-chatbot.js
- [X] T017 [US1] Add button auto-hide after 3 seconds if not clicked in selection-chatbot.js

## Phase 4: [US2] Mobile Text Selection

### Goal
Implement mobile long-press functionality to show the "Ask Chatbot" button.

**Independent Test Criteria**: When user long-presses on text for ~600ms on mobile, a "Ask Chatbot" button appears near the text.

- [X] T018 [US2] Implement touch event listeners for mobile text detection in selection-chatbot.js
- [X] T019 [US2] Add 600ms timer for long-press detection in selection-chatbot.js
- [X] T020 [US2] Calculate touch position for button placement in selection-chatbot.js
- [X] T021 [US2] Handle touch movement to cancel long-press detection in selection-chatbot.js
- [X] T022 [US2] Position button appropriately for mobile viewport in selection-chatbot.js
- [X] T023 [US2] Optimize button for touch targets (minimum 44px) in selection-chatbot.js
- [X] T024 [US2] Test mobile functionality across different screen sizes in selection-chatbot.js

## Phase 5: [US3] Chat Integration

### Goal
Integrate selected text with existing chatbot functionality.

**Independent Test Criteria**: Selected text is properly captured and sent to the backend, with response displayed in existing chat interface.

- [X] T025 [US3] Verify existing /selected_text endpoint in backend/app/main.py
- [X] T026 [US3] Implement postMessage communication with existing chat iframe in selection-chatbot.js
- [X] T027 [US3] Format selected text payload according to API contract in selection-chatbot.js
- [X] T028 [US3] Handle cases where no existing chat iframe exists in selection-chatbot.js
- [X] T029 [US3] Create temporary iframe if needed for chat communication in selection-chatbot.js
- [X] T030 [US3] Ensure button disappears after activation in selection-chatbot.js
- [X] T031 [US3] Test integration with existing floating chat widget in frontend/rag-chatbot-frontend/src/components/ChatWidget/

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Finalize implementation with error handling, performance optimization, and cross-browser compatibility.

- [X] T032 Add error handling for API communication failures in selection-chatbot.js
- [X] T033 Optimize performance to minimize DOM event impact in selection-chatbot.js
- [X] T034 Test cross-browser compatibility (Chrome, Firefox, Safari, Edge) in selection-chatbot.js
- [X] T035 Add keyboard shortcut (Ctrl/Cmd+Shift+C) support in selection-chatbot.js
- [X] T036 Implement proper cleanup of event listeners in selection-chatbot.js
- [X] T037 Add accessibility features (ARIA labels, focus management) in selection-chatbot.js
- [X] T038 Update documentation with usage instructions in specs/008-text-selection-chatbot/quickstart.md
- [X] T039 Test integration with existing Docusaurus plugins in frontend/rag-chatbot-frontend/src/plugins/
- [X] T040 Perform final testing across all device types and screen sizes