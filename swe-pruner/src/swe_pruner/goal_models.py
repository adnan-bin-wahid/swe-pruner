from pydantic import BaseModel, Field
from typing import List, Optional

class StructuredGoal(BaseModel):
    task_type: str = Field(description="The category of work, e.g., bug_fix, refactor, feature_addition, test_generation")
    objective: str = Field(description="Clear description of the precise goal focusing on the specific code target")
    identifiers: List[str] = Field(default_factory=list, description="Names of classes, functions, or variables referenced or required")
    observed_errors: List[str] = Field(default_factory=list, description="Diagnostic error messages or exception details observed in context")
    required_context: List[str] = Field(default_factory=list, description="Specific modules or dependencies that must be inspected")
    retrieval_questions: List[str] = Field(default_factory=list, description="Questions used for searching the codebase index")
    clarification_required: bool = Field(default=False, description="True if query is vague and editor evidence is insufficient")
