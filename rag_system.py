"""
RAG (Retrieval-Augmented Generation) System with Memory and Personas
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from datetime import datetime, timedelta
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Persona Definitions
class PersonaType(Enum):
    RESEARCHER = "researcher"
    EDUCATOR = "educator"
    STUDENT = "student"
    YOUTH_ADVOCATE = "youth_advocate"
    DATA_ANALYST = "data_analyst"
    POLICY_MAKER = "policy_maker"
    CUSTOM = "custom"

@dataclass
class Persona:
    """Define analysis personas with specific characteristics"""
    name: str
    type: PersonaType
    description: str
    focus_areas: List[str]
    analysis_style: str
    output_format: str
    key_questions: List[str]
    temperature: float = 0.7
    
    def to_prompt_context(self) -> str:
        """Convert persona to prompt context"""
        return f"""
        You are acting as a {self.name} ({self.type.value}).
        Description: {self.description}
        Focus Areas: {', '.join(self.focus_areas)}
        Analysis Style: {self.analysis_style}
        Output Format: {self.output_format}
        Key Questions to Consider: {'; '.join(self.key_questions)}
        """

class PersonaManager:
    """Manage different analysis personas"""
    
    def __init__(self):
        self.personas = self._initialize_personas()
        self.active_persona = None
    
    def _initialize_personas(self) -> Dict[str, Persona]:
        """Initialize default personas"""
        return {
            "researcher": Persona(
                name="Academic Researcher",
                type=PersonaType.RESEARCHER,
                description="Focused on rigorous analysis and evidence-based insights",
                focus_areas=["methodology", "validity", "reliability", "theoretical frameworks"],
                analysis_style="Systematic, thorough, citation-oriented",
                output_format="Academic paper style with references",
                key_questions=[
                    "What are the theoretical implications?",
                    "How does this relate to existing literature?",
                    "What are the methodological considerations?"
                ],
                temperature=0.3
            ),
            "educator": Persona(
                name="Youth Educator",
                type=PersonaType.EDUCATOR,
                description="Focused on learning outcomes and pedagogical applications",
                focus_areas=["learning objectives", "engagement", "accessibility", "scaffolding"],
                analysis_style="Clear, instructional, example-rich",
                output_format="Lesson plan format with activities",
                key_questions=[
                    "How can this be taught effectively?",
                    "What are the learning outcomes?",
                    "How can we engage youth with this content?"
                ],
                temperature=0.5
            ),
            "youth_advocate": Persona(
                name="Youth Advocate",
                type=PersonaType.YOUTH_ADVOCATE,
                description="Centered on youth voice, empowerment, and social justice",
                focus_areas=["youth voice", "empowerment", "equity", "action", "community"],
                analysis_style="Empowering, action-oriented, inclusive",
                output_format="Action plan with youth perspectives",
                key_questions=[
                    "How does this empower youth?",
                    "What actions can be taken?",
                    "Whose voices are represented?"
                ],
                temperature=0.6
            ),
            "data_analyst": Persona(
                name="Data Scientist",
                type=PersonaType.DATA_ANALYST,
                description="Focused on patterns, statistics, and quantitative insights",
                focus_areas=["patterns", "statistics", "trends", "correlations", "predictions"],
                analysis_style="Quantitative, precise, visual",
                output_format="Statistical report with visualizations",
                key_questions=[
                    "What patterns emerge from the data?",
                    "What are the statistical significance?",
                    "How can we visualize these insights?"
                ],
                temperature=0.2
            ),
            "policy_maker": Persona(
                name="Policy Advisor",
                type=PersonaType.POLICY_MAKER,
                description="Focused on policy implications and recommendations",
                focus_areas=["policy", "implementation", "stakeholders", "outcomes", "evaluation"],
                analysis_style="Strategic, pragmatic, evidence-based",
                output_format="Policy brief with recommendations",
                key_questions=[
                    "What are the policy implications?",
                    "Who are the stakeholders?",
                    "How can this be implemented?"
                ],
                temperature=0.4
            )
        }
    
    def set_active_persona(self, persona_key: str):
        """Set the active persona for analysis"""
        if persona_key in self.personas:
            self.active_persona = self.personas[persona_key]
            return True
        return False
    
    def get_active_persona(self) -> Optional[Persona]:
        """Get the currently active persona"""
        return self.active_persona
    
    def create_custom_persona(self, **kwargs) -> Persona:
        """Create a custom persona"""
        return Persona(type=PersonaType.CUSTOM, **kwargs)

class MemorySystem:
    """Long-term memory system for context retention"""
    
    def __init__(self, max_memories: int = 1000):
        self.memories = []
        self.max_memories = max_memories
        self.memory_index = {}
        self.session_context = []
        
    def add_memory(self, content: str, memory_type: str, metadata: Dict[str, Any] = None):
        """Add a memory to the system"""
        memory = {
            "id": hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:8],
            "content": content,
            "type": memory_type,
            "metadata": metadata or {},
            "timestamp": datetime.now(),
            "access_count": 0,
            "relevance_score": 1.0
        }
        
        self.memories.append(memory)
        self.memory_index[memory["id"]] = len(self.memories) - 1
        
        # Maintain memory limit
        if len(self.memories) > self.max_memories:
            self._prune_memories()
        
        return memory["id"]
    
    def _prune_memories(self):
        """Remove least relevant/oldest memories"""
        # Sort by relevance and recency
        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m["relevance_score"], m["timestamp"]),
            reverse=True
        )
        
        # Keep top memories
        self.memories = sorted_memories[:self.max_memories]
        
        # Rebuild index
        self.memory_index = {
            m["id"]: i for i, m in enumerate(self.memories)
        }
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant memories based on query"""
        if not self.memories:
            return []
        
        # Simple TF-IDF based retrieval
        vectorizer = TfidfVectorizer(max_features=100)
        memory_texts = [m["content"] for m in self.memories]
        
        try:
            tfidf_matrix = vectorizer.fit_transform(memory_texts + [query])
            query_vector = tfidf_matrix[-1]
            memory_vectors = tfidf_matrix[:-1]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, memory_vectors).flatten()
            
            # Get top-k memories
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            recalled_memories = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold for relevance
                    memory = self.memories[idx].copy()
                    memory["recall_score"] = float(similarities[idx])
                    memory["access_count"] += 1
                    recalled_memories.append(memory)
            
            return recalled_memories
        except:
            return []
    
    def get_session_context(self) -> str:
        """Get current session context"""
        return "\n".join(self.session_context[-10:])  # Last 10 context items
    
    def add_to_session(self, content: str):
        """Add to current session context"""
        self.session_context.append(f"[{datetime.now().strftime('%H:%M')}] {content}")
        if len(self.session_context) > 50:
            self.session_context = self.session_context[-50:]

class VectorStore:
    """Vector database for efficient similarity search"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        # Using IndexFlatIP for inner product (similar to cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents with their embeddings to the vector store"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        return len(self.documents)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        # Save index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        # Load index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.dimension = data['dimension']

class RAGSystem:
    """Main RAG system integrating retrieval, memory, and personas"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.memory_system = MemorySystem()
        self.persona_manager = PersonaManager()
        self.conversation_history = []
        
    def add_document(self, content: str, embedding: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a document to the RAG system"""
        # Add to vector store
        self.vector_store.add_documents(
            embeddings=embedding.reshape(1, -1),
            documents=[content],
            metadata=[metadata] if metadata else None
        )
        
        # Add to memory system
        self.memory_system.add_memory(
            content=content[:500],  # Store summary in memory
            memory_type="document",
            metadata=metadata
        )
        
        logger.info(f"Added document to RAG system")
    
    def retrieve_context(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant context for a query"""
        # Search in vector store
        vector_results = self.vector_store.search(query_embedding, top_k)
        
        # Recall from memory
        memory_results = self.memory_system.recall(query, top_k)
        
        # Get session context
        session_context = self.memory_system.get_session_context()
        
        # Get active persona
        persona = self.persona_manager.get_active_persona()
        
        return {
            "vector_results": vector_results,
            "memory_results": memory_results,
            "session_context": session_context,
            "persona": persona,
            "timestamp": datetime.now()
        }
    
    def generate_augmented_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an augmented prompt with retrieved context"""
        prompt_parts = []
        
        # Add persona context if available
        if context.get("persona"):
            prompt_parts.append(context["persona"].to_prompt_context())
        
        # Add retrieved documents
        if context.get("vector_results"):
            prompt_parts.append("\n## Relevant Documents:")
            for doc, score, meta in context["vector_results"][:3]:
                prompt_parts.append(f"- [{score:.2f}] {doc[:200]}...")
        
        # Add memory context
        if context.get("memory_results"):
            prompt_parts.append("\n## Related Memories:")
            for memory in context["memory_results"][:3]:
                prompt_parts.append(f"- [{memory['recall_score']:.2f}] {memory['content'][:150]}...")
        
        # Add session context
        if context.get("session_context"):
            prompt_parts.append(f"\n## Session Context:\n{context['session_context']}")
        
        # Add the actual query
        prompt_parts.append(f"\n## Current Query:\n{query}")
        
        # Add persona-specific instructions
        if context.get("persona"):
            persona = context["persona"]
            prompt_parts.append(f"\n## Instructions:\nRespond in the style of {persona.analysis_style}")
            prompt_parts.append(f"Focus on: {', '.join(persona.focus_areas)}")
            prompt_parts.append(f"Format: {persona.output_format}")
        
        return "\n".join(prompt_parts)
    
    def process_query(self, query: str, query_embedding: np.ndarray) -> Dict[str, Any]:
        """Process a query through the RAG system"""
        # Retrieve context
        context = self.retrieve_context(query, query_embedding)
        
        # Generate augmented prompt
        augmented_prompt = self.generate_augmented_prompt(query, context)
        
        # Add to conversation history
        self.conversation_history.append({
            "query": query,
            "context": context,
            "timestamp": datetime.now()
        })
        
        # Add to session memory
        self.memory_system.add_to_session(f"Query: {query[:100]}")
        
        return {
            "augmented_prompt": augmented_prompt,
            "context": context,
            "persona": context.get("persona"),
            "retrieved_docs": len(context.get("vector_results", [])),
            "memory_items": len(context.get("memory_results", []))
        }
    
    def update_memory_relevance(self, memory_id: str, relevance_delta: float):
        """Update memory relevance based on usage"""
        if memory_id in self.memory_system.memory_index:
            idx = self.memory_system.memory_index[memory_id]
            self.memory_system.memories[idx]["relevance_score"] += relevance_delta
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": self.vector_store.index.ntotal if self.vector_store.index else 0,
            "total_memories": len(self.memory_system.memories),
            "session_items": len(self.memory_system.session_context),
            "active_persona": self.persona_manager.active_persona.name if self.persona_manager.active_persona else None,
            "conversation_length": len(self.conversation_history)
        }

# Streamlit UI Components for RAG System
def render_rag_interface():
    """Render RAG system interface in Streamlit"""
    st.markdown("## 🧠 Intelligent Analysis (RAG)")
    
    # Initialize RAG system in session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    rag = st.session_state.rag_system
    
    # Persona selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👤 Select Analysis Persona")
        persona_options = list(rag.persona_manager.personas.keys())
        selected_persona = st.selectbox(
            "Choose persona for analysis",
            persona_options,
            format_func=lambda x: rag.persona_manager.personas[x].name
        )
        
        if st.button("Activate Persona"):
            rag.persona_manager.set_active_persona(selected_persona)
            st.success(f"✅ Activated: {rag.persona_manager.personas[selected_persona].name}")
    
    with col2:
        # Display active persona info
        if rag.persona_manager.active_persona:
            persona = rag.persona_manager.active_persona
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #003262 0%, #3B7EA1 100%); 
                        padding: 1rem; border-radius: 10px; color: white;">
                <h4 style="color: #FDB515; margin: 0;">Active Persona</h4>
                <p style="margin: 0.5rem 0;">{persona.name}</p>
                <small>{persona.description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Memory status
    st.markdown("### 💾 Memory System")
    stats = rag.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", stats["total_documents"])
    with col2:
        st.metric("Memories", stats["total_memories"])
    with col3:
        st.metric("Session Items", stats["session_items"])
    with col4:
        st.metric("Conversations", stats["conversation_length"])
    
    # Query interface
    st.markdown("### 🔍 Intelligent Query")
    query = st.text_area("Enter your question or analysis request:", height=100)
    
    if st.button("🚀 Analyze with RAG", type="primary"):
        if query:
            with st.spinner("Processing with RAG system..."):
                # Generate simple embedding for demo
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=384)
                try:
                    embedding = vectorizer.fit_transform([query]).toarray()[0]
                    
                    # Process through RAG
                    result = rag.process_query(query, embedding)
                    
                    # Display results
                    st.success(f"✅ Retrieved {result['retrieved_docs']} documents and {result['memory_items']} memories")
                    
                    with st.expander("View Augmented Context"):
                        st.text(result["augmented_prompt"])
                    
                    # Add to memory
                    rag.memory_system.add_memory(
                        content=query,
                        memory_type="query",
                        metadata={"persona": selected_persona}
                    )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a query")
    
    # Display recent memories
    with st.expander("Recent Memories"):
        recent_memories = rag.memory_system.memories[-5:]
        for memory in reversed(recent_memories):
            st.write(f"**[{memory['type']}]** {memory['content'][:100]}...")
            st.caption(f"Added: {memory['timestamp'].strftime('%Y-%m-%d %H:%M')}")

def render_persona_builder():
    """Interface for creating custom personas"""
    st.markdown("### 🎭 Create Custom Persona")
    
    with st.form("custom_persona"):
        name = st.text_input("Persona Name")
        description = st.text_area("Description")
        focus_areas = st.text_input("Focus Areas (comma-separated)").split(",")
        analysis_style = st.text_input("Analysis Style")
        output_format = st.text_input("Output Format")
        key_questions = st.text_area("Key Questions (one per line)").split("\n")
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.5)
        
        if st.form_submit_button("Create Persona"):
            if name and description:
                # Create custom persona
                persona = Persona(
                    name=name,
                    type=PersonaType.CUSTOM,
                    description=description,
                    focus_areas=[f.strip() for f in focus_areas],
                    analysis_style=analysis_style,
                    output_format=output_format,
                    key_questions=[q.strip() for q in key_questions if q.strip()],
                    temperature=temperature
                )
                
                # Add to persona manager
                if 'rag_system' in st.session_state:
                    st.session_state.rag_system.persona_manager.personas[name.lower().replace(" ", "_")] = persona
                    st.success(f"✅ Created persona: {name}")
            else:
                st.error("Please fill in required fields")