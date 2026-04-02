# Research Loop Rules (Daily Category Rotation)

## Purpose

Aggressively fill the awesome-opensource-ai list by targeting one category per day with deep research.

## Frequency

Run daily at 9:00 AM Europe/Amsterdam. Each run targets ONE specific category (rotating 14-day cycle).

## Category Rotation Schedule

| Day | Category | Target Section |
|-----|----------|----------------|
| Day 1 (Apr 3) | Core Frameworks & Libraries | §1 |
| Day 2 (Apr 4) | Open Foundation Models | §2 |
| Day 3 (Apr 5) | Inference Engines & Serving | §3 |
| Day 4 (Apr 6) | Agentic AI & Multi-Agent Systems | §4 |
| Day 5 (Apr 7) | RAG & Knowledge | §5 |
| Day 6 (Apr 8) | Generative Media Tools | §6 |
| Day 7 (Apr 9) | Training & Fine-tuning | §7 |
| Day 8 (Apr 10) | MLOps / LLMOps | §8 |
| Day 9 (Apr 11) | Evaluation & Benchmarks | §9 |
| Day 10 (Apr 12) | AI Safety & Interpretability | §10 |
| Day 11 (Apr 13) | Specialized Domains | §11 |
| Day 12 (Apr 14) | User Interfaces & Platforms | §12 |
| Day 13 (Apr 15) | Developer Tools | §13 |
| Day 14 (Apr 16) | Resources & Learning | §14 |
| Day 15+ | Cycle repeats | - |

## Today's Target Category

Check the current date and pick the matching category from the rotation above.

## Deep Research Protocol (Per Category)

Don't just scan trending. Do targeted discovery for the specific category:

### Category-Specific Sources

**§1 Core Frameworks:**
- PyTorch ecosystem repos (torchvision, torchaudio, torchtext)
- JAX/Flax adjacent tools
- Hugging Face ecosystem beyond transformers
- Data processing libraries (beyond pandas/polars)
- New ML frameworks (Julia, Rust, C++)

**§2 Foundation Models:**
- Hugging Face model hub trending
- LLM leaderboard new entries
- Paper releases with open weights
- Regional model labs (Europe, Asia)
- Small/edge models (<7B params)

**§3 Inference & Serving:**
- vLLM forks and alternatives
- New quantization methods
- Edge/mobile inference tools
- Kubernetes operators for LLMs
- New GGUF/ONNX tooling

**§4 Agentic AI:**
- New agent frameworks (check GitHub tags: agent, autonomous)
- Multi-agent orchestration tools
- Agent memory systems
- Tool-use libraries
- Agent eval benchmarks

**§5 RAG & Knowledge:**
- Vector DBs beyond Chroma/Qdrant/Pinecone
- New embedding models
- Document parsers
- Graph RAG tools
- Hybrid search systems

**§6 Generative Media:**
- New diffusion models (image/video/3D)
- Audio generation (music, speech, SFX)
- Animation tools
- Real-time generation systems
- LoRA/training tools for media

**§7 Training & Fine-tuning:**
- New training frameworks
- RLHF tools
- Fine-tuning UIs
- Dataset curation tools
- Synthetic data generators

**§8 MLOps/LLMOps:**
- Prompt management tools
- LLM observability
- Cost tracking
- A/B testing for models
- Feature stores for ML

**§9 Evaluation:**
- New benchmarks
- Evaluation harnesses
- Red-teaming tools
- Model comparison UIs
- Safety eval suites

**§10 Safety & Alignment:**
- Interpretability tools
- Alignment training methods
- Adversarial testing
- Constitutional AI tools
- Guardrails frameworks

**§11 Specialized Domains:**
- Scientific AI (protein, drug, materials)
- Code-specific tools
- Legal/medical/finance AI
- Robotics simulators
- Game AI

**§12 User Interfaces:**
- New ChatGPT alternatives (self-hosted)
- Mobile apps
- Voice interfaces
- Desktop assistants
- Browser extensions

**§13 Developer Tools:**
- IDE plugins beyond Continue/Cline
- Testing frameworks for AI
- Documentation generators
- API clients
- CLI tools

**§14 Resources:**
- New courses/bootcamps
- Active communities
- Newsletters/podcasts
- Benchmark leaderboards
- Model cards repositories

## Qualification Criteria

For the target category specifically:

- OSI-approved license
- Active development (commits within 3 months)
- Quality signal: >500 GitHub stars OR featured in notable publication OR backed by credible org
- Not already in the list
- Fits the target category (don't add RAG tools to Agentic section)

## Output: Single Category-Focused PR

**One PR per day** targeting today's category only.

Format:
```
[Research] Add N entries to {Category} — {Date}
```

PR body template:
```markdown
## Category: {Category}

Adding {N} qualifying projects discovered through targeted research.

| Project | Stars | License | Why It Fits |
|---------|-------|---------|-------------|
| {name} | {stars} | {license} | {one-line justification} |

### Research Sources
- Source 1: {what was checked}
- Source 2: {what was checked}
- Source 3: {what was checked}

### Verification
All projects checked for:
- [x] OSI-approved license
- [x] Active development
- [x] >500 stars or notable backing
- [x] Not already in README
- [x] Fits {Category} section
```

## Limits

- Max **5 entries per category per day**
- If <3 qualifying projects found, still open PR with what you found
- If 0 qualifying projects found, skip opening PR (post checkin noting "no qualifying projects found for {Category} today")

## Edge Cases

- **Project overlaps multiple categories:** Pick the best fit, note overlap in PR body
- **Category already has 50+ entries:** Be more selective (focus on newest/highest quality)
- **Duplicate found during research:** Note it in PR body, pick the better one
- **Stars <500 but clearly significant:** Include with justification (e.g., "just launched, backed by Anthropic")

## Remember

This is a **14-day cycle**. By April 17, every category should have been deeply researched once. Then we cycle again with fresh eyes and new projects that emerged.
