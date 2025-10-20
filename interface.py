import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import spacy

# =====================================================
# 🔹 0️⃣ Setup spaCy model (auto fallback)
# =====================================================
@st.cache_resource
def load_spacy():
    """Load spaCy model dengan fallback otomatis."""
    try:
        return spacy.load("en_core_web_md")  # model besar dengan word vectors
    except OSError:
        st.warning("⚠️ Model 'en_core_web_md' tidak ditemukan. Menggunakan 'en_core_web_sm' sebagai fallback.")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =====================================================
# 1️⃣ Load dataset SQuAD
# =====================================================
@st.cache_data(show_spinner=False)
def load_squad(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        contexts, questions = [], []
        for article in data['data']:
            for p in article['paragraphs']:
                contexts.append(p['context'])
                for qa in p['qas']:
                    questions.append(qa['question'])
        return contexts, questions
    except FileNotFoundError:
        st.error(f"❌ File {file_path} tidak ditemukan!")
        return [], []

contexts, questions = load_squad("dev-v1.1.json")

# =====================================================
# 2️⃣ Load embedding & reranker models
# =====================================================
@st.cache_resource(show_spinner=True)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedder, reranker = load_models()

# Encode sebagian context agar lebih ringan
@st.cache_resource(show_spinner=True)
def encode_contexts(contexts):
    if not contexts:
        return None
    return embedder.encode(contexts[:200], convert_to_tensor=True)

context_embeddings = encode_contexts(contexts)
sample_contexts = contexts[:200] if contexts else []

# =====================================================
# 🔍 Simple Synonym Expansion (Static + Semantic)
# =====================================================
# Dictionary sinonim sederhana
SIMPLE_SYNONYMS = {
    'who': ['person', 'people', 'individual'],
    'what': ['thing', 'which'],
    'when': ['time', 'date'],
    'where': ['place', 'location'],
    'why': ['reason', 'cause'],
    'how': ['method', 'way', 'manner'],
    'won': ['victory', 'win', 'champion', 'winner', 'defeated', 'beat'],
    'super': ['great', 'excellent'],
    'bowl': ['game', 'championship'],
    'car': ['vehicle', 'automobile'],
    'big': ['large', 'huge', 'enormous'],
    'small': ['tiny', 'little', 'compact'],
    'good': ['great', 'excellent', 'fine'],
    'bad': ['poor', 'terrible', 'awful'],
}

def get_semantic_synonyms(word, nlp, top_n=5):
    """Cari sinonim semantik berbasis similarity spaCy."""
    if not nlp.vocab[word].has_vector:
        return []
    target = nlp(word)
    sims = []
    for w in nlp.vocab:
        if w.is_lower and w.has_vector and w.is_alpha:
            sim = target.similarity(w)
            if sim > 0.65 and w.text != word:
                sims.append((w.text, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in sims[:top_n]]

def expand_with_synonyms(query):
    """Ekspansi query dengan sinonim statis dan semantik."""
    words = query.lower().split()
    expanded = set(words)
    
    for word in words:
        clean_word = word.strip('.,!?;:')
        
        # Sinonim statis
        if clean_word in SIMPLE_SYNONYMS:
            expanded.update(SIMPLE_SYNONYMS[clean_word])
        
        # Sinonim semantik (kalau model punya vektor)
        expanded.update(get_semantic_synonyms(clean_word, nlp))
    
    return " ".join(expanded)

# =====================================================
# 🧩 Entity Recognition (NER)
# =====================================================
def extract_entities(query):
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# =====================================================
# 🎨 Query Enhancement dengan Lemmatization
# =====================================================
def enhance_query(query):
    """Tambahkan lemma untuk meningkatkan matching."""
    doc = nlp(query)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(set(lemmas + [query]))

# =====================================================
# 3️⃣ Streamlit UI
# =====================================================
st.title("🦉 Hummingbird NLP – Semantic Search Plus")
st.markdown("""
Implementasi **algoritma Google Hummingbird** dengan fitur tambahan:
- 🔤 *Query Enhancement* (lemmatization via spaCy)
- 🧠 *Entity Recognition* (memahami entitas penting)
- 🎯 *Reranking* hasil menggunakan model cross-encoder
- 🔍 *Semantic Embedding* untuk memahami makna kalimat
""")

if not contexts:
    st.warning("⚠️ Dataset SQuAD tidak berhasil dimuat. Pastikan file 'dev-v1.1.json' ada di folder yang sama.")
    st.stop()

query = st.text_input("Masukkan pertanyaan (Bahasa Inggris):", placeholder="e.g. Who won Super Bowl 50?")

if st.button("Cari Makna 🔍") or query:
    if not query.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu!")
    else:
        with st.spinner("Memproses makna pertanyaan..."):

            # ---- (1) Query enhancement
            enhanced_query = enhance_query(query)
            expanded_query = expand_with_synonyms(query)
            final_query = f"{query} {enhanced_query} {expanded_query}"

            # ---- (2) Entity recognition
            entities = extract_entities(query)

            # ---- (3) Embedding similarity search
            query_emb = embedder.encode(final_query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_emb, context_embeddings)[0]
            top_indices = cosine_scores.argsort(descending=True)[:5]
            top_contexts = [sample_contexts[i] for i in top_indices]
            top_scores = [cosine_scores[i].item() for i in top_indices]

            # ---- (4) Reranking step
            pairs = [[query, ctx] for ctx in top_contexts]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(zip(top_contexts, top_scores, rerank_scores), key=lambda x: x[2], reverse=True)

            best_context, base_score, rerank_score = ranked[0]

        # =====================================================
        # 💡 Display Results
        # =====================================================
        st.subheader("💬 Query yang diproses:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Query asli:**")
            st.code(query, language=None)
        with col2:
            st.write(f"**Enhanced query:**")
            st.code(enhanced_query[:100] + "...", language=None)

        if entities:
            st.write("**🏷️ Entitas yang dikenali:**")
            entity_cols = st.columns(len(entities))
            for idx, (text, label) in enumerate(entities):
                with entity_cols[idx]:
                    st.metric(label=label, value=text)

        st.markdown("---")
        st.subheader("🎯 Hasil Semantic Search + Reranking")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cosine Similarity", f"{base_score:.3f}")
        with col2:
            st.metric("Rerank Score", f"{rerank_score:.3f}")
        with col3:
            improvement = ((rerank_score - base_score) / base_score * 100) if base_score > 0 else 0
            st.metric("Improvement", f"{improvement:+.1f}%")
        
        st.write("**📘 Konteks paling relevan:**")
        st.text_area("", best_context, height=200, label_visibility="collapsed")

        st.subheader("📚 Top 3 hasil setelah reranking:")
        for i, (ctx, base, re) in enumerate(ranked[:3], 1):
            with st.expander(f"**#{i}** - Cosine: {base:.3f} | Rerank: {re:.3f}"):
                st.write(ctx[:500] + ("..." if len(ctx) > 500 else ""))

# =====================================================
# Sidebar Info
# =====================================================
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini meniru **Google Hummingbird (NLP)**:

**Fitur:**
- ✅ **Semantic Embedding** dengan SentenceTransformer  
- ✅ **Query Enhancement** via spaCy Lemmatization  
- ✅ **Entity Recognition** untuk memahami konteks  
- ✅ **Cross-Encoder Reranking** untuk hasil terbaik  
- ✅ **Sinonim Semantik** berbasis spaCy word vectors  

Tanpa NLTK WordNet – seluruh NLP diproses menggunakan spaCy.
""")

st.sidebar.markdown("---")
st.sidebar.write("**📊 Statistik Dataset:**")
st.sidebar.metric("Total Contexts", len(contexts))
st.sidebar.metric("Sample Used", len(sample_contexts))
st.sidebar.metric("Total Questions", len(questions))
st.sidebar.markdown("---")
st.sidebar.info("💡 Gunakan pertanyaan dalam Bahasa Inggris untuk hasil terbaik!")
