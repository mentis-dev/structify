#!/usr/bin/env python3
import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from core.utils import init_model

@dataclass
class FeedbackRecord:
    """Structure for storing feedback records"""
    id: str
    extraction_type: str  # 'stakeholder', 'factor', 'pain_point'
    item_id: str
    original_classification: Dict[str, Any]
    user_agrees: bool
    user_feedback: str
    user_id: Optional[str]
    timestamp: str
    model_used: str
    correction_applied: bool = False

class FeedbackStorage:
    """Handles storage and retrieval of user feedback"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                extraction_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                original_classification TEXT NOT NULL,
                user_agrees BOOLEAN NOT NULL,
                user_feedback TEXT,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                model_used TEXT NOT NULL,
                correction_applied BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create training examples table for fine-tuning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                expected_output TEXT NOT NULL,
                feedback_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (feedback_id) REFERENCES feedback (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: FeedbackRecord) -> bool:
        """Store a feedback record in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (id, extraction_type, item_id, original_classification, user_agrees, 
                 user_feedback, user_id, timestamp, model_used, correction_applied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.id,
                feedback.extraction_type,
                feedback.item_id,
                json.dumps(feedback.original_classification),
                feedback.user_agrees,
                feedback.user_feedback,
                feedback.user_id,
                feedback.timestamp,
                feedback.model_used,
                feedback.correction_applied
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False
    
    def get_negative_feedback(self, extraction_type: Optional[str] = None) -> List[FeedbackRecord]:
        """Retrieve negative feedback for retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM feedback WHERE user_agrees = FALSE"
        params = []
        
        if extraction_type:
            query += " AND extraction_type = ?"
            params.append(extraction_type)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        feedback_records = []
        for row in rows:
            feedback_records.append(FeedbackRecord(
                id=row[0],
                extraction_type=row[1],
                item_id=row[2],
                original_classification=json.loads(row[3]),
                user_agrees=bool(row[4]),
                user_feedback=row[5],
                user_id=row[6],
                timestamp=row[7],
                model_used=row[8],
                correction_applied=bool(row[9])
            ))
        
        return feedback_records
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_agrees = TRUE")
        positive_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_agrees = FALSE")
        negative_feedback = cursor.fetchone()[0]
        
        # By extraction type
        cursor.execute('''
            SELECT extraction_type, 
                   COUNT(*) as total,
                   SUM(CASE WHEN user_agrees THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN NOT user_agrees THEN 1 ELSE 0 END) as negative
            FROM feedback 
            GROUP BY extraction_type
        ''')
        by_type = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "accuracy_rate": positive_feedback / total_feedback if total_feedback > 0 else 0,
            "by_type": {row[0]: {"total": row[1], "positive": row[2], "negative": row[3]} for row in by_type}
        }

class FeedbackLearningAgent:
    """Agent that learns from user feedback to improve classifications"""
    
    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        feedback_storage: Optional[FeedbackStorage] = None
    ):
        self.model = model
        self.feedback_storage = feedback_storage or FeedbackStorage()
        self.correction_examples = []
        self._load_correction_examples()
    
    def _load_correction_examples(self):
        """Load correction examples from negative feedback"""
        negative_feedback = self.feedback_storage.get_negative_feedback()
        
        for feedback in negative_feedback:
            if feedback.user_feedback.strip():  # Only use feedback with explanations
                self.correction_examples.append({
                    "original": feedback.original_classification,
                    "feedback": feedback.user_feedback,
                    "type": feedback.extraction_type
                })
    
    async def classify_with_feedback(
        self,
        text: str,
        extraction_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform classification using learned feedback patterns
        
        Args:
            text: Text to analyze
            extraction_type: Type of extraction ('stakeholder', 'factor', 'pain_point')
            context: Additional context for classification
            
        Returns:
            Classification results with confidence and reasoning
        """
        # Get relevant correction examples for this extraction type
        relevant_examples = [ex for ex in self.correction_examples if ex["type"] == extraction_type]
        
        # Build prompt with feedback examples
        prompt = self._build_feedback_aware_prompt(text, extraction_type, relevant_examples)
        
        # Initialize model and make prediction
        model = init_model({"configurable": {"model": self.model}})
        messages = [HumanMessage(content=prompt)]
        response = await model.ainvoke(messages)
        
        try:
            # Parse response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            result = json.loads(content)
            result["feedback_examples_used"] = len(relevant_examples)
            result["model_used"] = self.model
            
            return result
            
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse classification result",
                "raw_response": response.content,
                "feedback_examples_used": len(relevant_examples)
            }
    
    def _build_feedback_aware_prompt(
        self,
        text: str,
        extraction_type: str,
        examples: List[Dict[str, Any]]
    ) -> str:
        """Build a prompt that incorporates user feedback examples"""
        
        base_prompts = {
            "stakeholder": """
            You are an expert stakeholder analyst. Classify the following stakeholder:
            
            Categories: Regulator, Supplier, Consumer, Competitor, Partner, Influencer, Internal
            Hierarchy Levels: Macro, Meso, Micro
            """,
            "factor": """
            You are an expert factor analyst. Classify the following factor:
            
            Types: Indirect Driver, Exogenous Driver, Direct Driver, Outcome
            """,
            "pain_point": """
            You are an expert pain point analyst. Identify and classify the following pain point:
            
            Focus on clear, actionable pain points with specific categories.
            """
        }
        
        prompt = base_prompts.get(extraction_type, "Analyze the following:")
        
        # Add feedback examples if available
        if examples:
            prompt += "\n\nIMPORTANT: Learn from these previous corrections:\n"
            
            for i, example in enumerate(examples[:5]):  # Limit to 5 examples
                prompt += f"\nCorrection Example {i+1}:\n"
                prompt += f"Original Classification: {json.dumps(example['original'], indent=2)}\n"
                prompt += f"User Feedback: {example['feedback']}\n"
                prompt += "---"
            
            prompt += "\nPlease consider these corrections when making your classification.\n"
        
        prompt += f"\n\nText to analyze:\n{text}\n\n"
        prompt += "Provide your analysis as JSON with reasoning for your decisions."
        
        return prompt
    
    def store_user_feedback(
        self,
        item_id: str,
        extraction_type: str,
        original_classification: Dict[str, Any],
        user_agrees: bool,
        user_feedback: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Store user feedback for future learning"""
        
        feedback_record = FeedbackRecord(
            id=f"{extraction_type}_{item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            extraction_type=extraction_type,
            item_id=item_id,
            original_classification=original_classification,
            user_agrees=user_agrees,
            user_feedback=user_feedback,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            model_used=self.model
        )
        
        success = self.feedback_storage.store_feedback(feedback_record)
        
        if success and not user_agrees:
            # Reload correction examples to include new feedback
            self._load_correction_examples()
        
        return success
    
    async def batch_reclassify(
        self,
        items: List[Dict[str, Any]],
        extraction_type: str
    ) -> List[Dict[str, Any]]:
        """
        Reclassify a batch of items using learned feedback
        
        Args:
            items: List of items to reclassify
            extraction_type: Type of extraction
            
        Returns:
            List of reclassified items with comparison to originals
        """
        reclassified = []
        
        for item in items:
            # Extract text for reclassification (this depends on your data structure)
            text = self._extract_text_for_reclassification(item, extraction_type)
            
            if text:
                new_classification = await self.classify_with_feedback(text, extraction_type)
                
                reclassified.append({
                    "original": item,
                    "reclassified": new_classification,
                    "changes_detected": self._detect_changes(item, new_classification)
                })
            else:
                reclassified.append({
                    "original": item,
                    "reclassified": item,
                    "changes_detected": False,
                    "error": "Could not extract text for reclassification"
                })
        
        return reclassified
    
    def _extract_text_for_reclassification(self, item: Dict[str, Any], extraction_type: str) -> Optional[str]:
        """Extract relevant text from an item for reclassification"""
        if extraction_type == "stakeholder":
            name = item.get("name", "")
            role = item.get("role", "")
            return f"Name: {name}\nRole: {role}"
        elif extraction_type == "factor":
            label = item.get("Label", "")
            description = item.get("Description", "")
            return f"Factor: {label}\nDescription: {description}"
        elif extraction_type == "pain_point":
            category = item.get("category", "")
            description = item.get("description", "")
            return f"Category: {category}\nDescription: {description}"
        
        return None
    
    def _detect_changes(self, original: Dict[str, Any], new: Dict[str, Any]) -> bool:
        """Detect if there are significant changes between classifications"""
        key_fields = {
            "stakeholder": ["category", "hierarchy_level"],
            "factor": ["Type"],
            "pain_point": ["category"]
        }
        
        # Simple change detection - compare key fields
        for field_set in key_fields.values():
            for field in field_set:
                if original.get(field) != new.get(field):
                    return True
        
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report based on feedback"""
        stats = self.feedback_storage.get_feedback_stats()
        
        # Calculate improvement metrics
        negative_feedback = self.feedback_storage.get_negative_feedback()
        common_errors = {}
        
        for feedback in negative_feedback:
            error_type = f"{feedback.extraction_type}_{feedback.original_classification.get('category', 'unknown')}"
            if error_type not in common_errors:
                common_errors[error_type] = []
            common_errors[error_type].append(feedback.user_feedback)
        
        return {
            "overall_stats": stats,
            "common_errors": {k: len(v) for k, v in common_errors.items()},
            "correction_examples_loaded": len(self.correction_examples),
            "recommendations": self._generate_recommendations(stats, common_errors)
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any], common_errors: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations for improving the system"""
        recommendations = []
        
        accuracy_rate = stats.get("accuracy_rate", 0)
        
        if accuracy_rate < 0.7:
            recommendations.append("Consider fine-tuning the model with collected feedback data")
        
        if stats.get("negative_feedback", 0) > 10:
            recommendations.append("Implement active learning to focus on difficult cases")
        
        # Analyze common error patterns
        for error_type, count in sorted(common_errors.items(), key=lambda x: x[1], reverse=True):
            if count > 3:
                recommendations.append(f"Review classification rules for {error_type} - {count} similar errors found")
        
        if len(recommendations) == 0:
            recommendations.append("System performance is good. Continue collecting feedback for further improvements.")
        
        return recommendations

# Integration functions for your Streamlit app
def initialize_feedback_system(db_path: str = "feedback.db") -> FeedbackLearningAgent:
    """Initialize the feedback learning system"""
    storage = FeedbackStorage(db_path)
    agent = FeedbackLearningAgent(feedback_storage=storage)
    return agent

def process_streamlit_feedback(
    agent: FeedbackLearningAgent,
    user_feedback_dict: Dict[str, Dict[str, Any]],
    extraction_results: Dict[str, Any]
) -> bool:
    """Process feedback from Streamlit session state"""
    
    success_count = 0
    
    # Process stakeholder feedback
    for item_id, feedback in user_feedback_dict.get("stakeholders", {}).items():
        if not feedback.get("agrees", True):
            # Find the original stakeholder
            stakeholder = None
            for s in extraction_results.get("stakeholders", []):
                if s.get("name") == item_id:
                    stakeholder = s
                    break
            
            if stakeholder:
                success = agent.store_user_feedback(
                    item_id=item_id,
                    extraction_type="stakeholder",
                    original_classification=stakeholder,
                    user_agrees=False,
                    user_feedback=feedback.get("feedback", "")
                )
                if success:
                    success_count += 1
    
    # Process factor and pain point feedback similarly...
    # (Follow the same pattern for factors and pain_points)
    
    return success_count > 0

# Example usage for batch reprocessing
async def reprocess_with_feedback(
    agent: FeedbackLearningAgent,
    extraction_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Reprocess extraction results using learned feedback"""
    
    improved_results = extraction_results.copy()
    
    # Reclassify stakeholders if there's negative feedback
    if extraction_results.get("stakeholders"):
        stakeholder_feedback = agent.feedback_storage.get_negative_feedback("stakeholder")
        
        if stakeholder_feedback:
            print(f"Reclassifying {len(extraction_results['stakeholders'])} stakeholders using {len(stakeholder_feedback)} feedback examples")
            
            reclassified = await agent.batch_reclassify(
                extraction_results["stakeholders"],
                "stakeholder"
            )
            
            # Update results with improved classifications
            improved_stakeholders = []
            for result in reclassified:
                if result.get("changes_detected"):
                    print(f"Updated classification for: {result['original'].get('name')}")
                    improved_stakeholders.append(result["reclassified"])
                else:
                    improved_stakeholders.append(result["original"])
            
            improved_results["stakeholders"] = improved_stakeholders
    
    return improved_results
