name: Reviewer Feedback
description: Template for JOSS reviewers to provide comments and suggestions
title: "[Review] "
labels: ["review"]
body:
  - type: markdown
    attributes:
      value: |
        Thank you for reviewing this submission! Please use the sections below to share your feedback.
  - type: textarea
    id: strengths
    attributes:
      label: Strengths
      description: What aspects of the software and paper do you find strong or well-done?
  - type: textarea
    id: improvements
    attributes:
      label: Suggestions for improvement
      description: Are there areas that could be clarified, improved, or expanded?
  - type: textarea
    id: technical
    attributes:
      label: Technical checks
      description: Any issues with installation, functionality, or reproducibility?
  - type: textarea
    id: other
    attributes:
      label: Additional comments
      description: Anything else you'd like to add.
