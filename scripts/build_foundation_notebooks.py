"""Author the six M1 notebooks. Narrative and experiment cells are source-controlled here."""

import hashlib
import subprocess
import textwrap

import nbformat as nbf

from vector_masterclass.config import root


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    source = textwrap.dedent(text).strip()
    formatted = subprocess.run(
        [str(root() / ".venv/bin/ruff"), "format", "--stdin-filename", "cell.py", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return nbf.v4.new_code_cell(formatted.strip())


def notebook(id_, title, profile, prerequisites, cells, *, docker=None):
    if docker is None:
        docker = id_ not in {"01", "02", "03", "04"}
    setup = code(f'''
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from vector_masterclass.lesson import LessonRun
        from vector_masterclass.config import root

        plt.rcParams.update({{"figure.figsize": (8, 4), "axes.spines.top": False, "axes.spines.right": False}})
        lesson = LessonRun("{id_}", "{profile}", docker={docker})
        print("Kernel:", sys.executable)
        print("Workspace:", root())
    ''')
    references = md("""
    ## References and next steps
    Consult the [Qdrant documentation](https://qdrant.tech/documentation/),
    the [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
    and [uv's Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/).
    The exact pinned configuration is in `configs/models.json`; worked exercises are in `docs/SOLUTIONS.md`.
    This notebook's measurements apply to the recorded local run, not a production deployment.
    """)
    nb = nbf.v4.new_notebook(
        cells=[md(f"# {id_} · {title}"), cells[0], setup, *cells[1:], references]
    )
    for index, cell in enumerate(nb.cells):
        cell.id = hashlib.sha256(f"{id_}:{index}:{cell.source}".encode()).hexdigest()[:12]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Vector Masterclass (uv)",
            "language": "python",
            "name": "vector-masterclass",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "course": {
            "id": id_,
            "profile": profile,
            "prerequisites": prerequisites,
            "timeout_seconds": {"T": 120, "B": 600, "G": 1200, "O": 3600}[profile],
            "live_required": True,
            "optional": id_ == "27b",
            "platform": "macOS Apple Silicon",
            "resource_contract": "docs/LESSON_CONTRACTS.md",
        },
    }
    folder = root() / "notebooks"
    folder.mkdir(exist_ok=True)
    name = title.lower().replace(" ", "_").replace("/", "_")
    nbf.write(nb, folder / f"{id_}_{name}.ipynb")


def main():
    notebook(
        "00",
        "Environment and the first retrieval",
        "T",
        [],
        [
            md("""
        ## The engineering question
        Can we prove that this notebook uses the intended Python environment, local model artifacts, and an actual Qdrant server?

        **Learning outcomes:** identify each execution boundary; explain why native model inference and Docker storage are separate; perform a real insert/query; distinguish server readiness from index readiness.

        Run `uv sync --locked`, `docker compose up -d`, and `uv run vmc bootstrap` first. Model downloads are explicit setup work. Inference never silently downloads a different model or substitutes example results. The course supports macOS on Apple Silicon; native PyTorch uses CPU or MPS, while Qdrant runs on CPU inside Docker's Linux VM.
        """),
            code("""
        import json
        from vector_masterclass.preflight import preflight
        from vector_masterclass.config import model_config, model_path
        environment = preflight()
        display(environment)
        rows = []
        for key in ['dense', 'reranker', 'sparse', 'generator']:
            manifest = json.loads((model_path(key) / 'manifest.json').read_text())
            rows.append({'model': key, 'device': environment['inference_devices'][key], 'artifact_MiB': manifest['bytes'] / 2**20})
        models = lesson.table(pd.DataFrame(rows), 'models')
        display(models)
        fig, ax = plt.subplots()
        ax.bar(models.model, models.artifact_MiB, color='#2563eb')
        ax.set(ylabel='Pinned model files (MiB)', title='Download/storage cost is distinct from runtime memory')
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## A real round trip
        We create an isolated collection, embed two sentences, write them, and query the server. `wait=True` waits for the write operation; it does not by itself prove that every segment has an HNSW index. The tiny collection is an API smoke check, not an ANN benchmark.
        """),
            code("""
        from uuid import uuid4
        from qdrant_client import QdrantClient, models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.models import embed
        texts = ['Atlas warranty lasts two years.', 'Boreal warranty lasts five years.']
        vectors = embed(texts)
        client = QdrantClient(url=qdrant_url(), timeout=10)
        collection = 'vmc_00_' + uuid4().hex[:8]
        client.create_collection(collection, vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE))
        try:
            client.upsert(collection, points=[qm.PointStruct(id=i, vector=v.tolist(), payload={'text': texts[i]}) for i, v in enumerate(vectors)], wait=True)
            hits = client.query_points(collection, query=embed(['Atlas warranty'], role='query')[0].tolist(), limit=2).points
            display(lesson.table(pd.DataFrame([{'id': h.id, 'score': h.score, 'text': h.payload['text']} for h in hits]), 'roundtrip'))
            first_hit = hits[0].id
        finally:
            client.delete_collection(collection)
            client.close()
        """),
            md("""
        ## Failure, exercise, and production implications
        Stop this project's Qdrant service and rerun the readiness cell: it must fail within its timeout. A missing model must point to the bootstrap step instead of reaching the internet. Restart Qdrant before continuing.

        **Exercise:** explain why moving the Python process into this Linux container would change access to the Mac's Metal GPU. Add a third sentence and predict the best result before querying.

        In production, readiness, persistence, authentication, and index readiness are different checks. This local smoke test establishes none of the deployment's availability guarantees. Use this setup for learning and controlled experiments; use the operations lessons before interpreting it as an operational design. The cleanup removes only this notebook's randomly named collection.
        """),
            code("""
        manifest = lesson.finish({'real_server': environment['qdrant']['version'] == '1.17.1', 'correct_top_hit': first_hit == 0, 'embedding_shape': vectors.shape == (2, 384), 'native_apple_silicon': environment['hardware']['supported_host']}, observations={'environment': environment})
        """),
        ],
    )

    notebook(
        "01",
        "Geometry and similarity",
        "T",
        ["00"],
        [
            md(r"""
        ## The engineering question
        When do dot product, cosine similarity, and Euclidean distance produce different neighbors?

        **Learning outcomes:** compute the scores by hand; explain the effect of vector magnitude; recognize undefined cosine for zero vectors; connect metric choice to model training.

        For vectors $q$ and $x$, the dot product is $q^T x$, cosine is $q^T x/(\|q\|\|x\|)$, and Euclidean distance is $\|q-x\|_2$. Larger dot/cosine scores rank first; smaller distances rank first. With unit vectors, squared Euclidean distance equals $2-2\cos(q,x)$, so their ordering agrees. Without normalization, it need not.
        """),
            code("""
        query = np.array([1., 0.])
        vectors = np.array([[1., 0.], [3., 1.], [0., 1.], [-1., 0.]])
        names = ['exact direction', 'large magnitude', 'orthogonal', 'opposite']
        dots = vectors @ query
        cosine = dots / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        distances = np.linalg.norm(vectors - query, axis=1)
        results = lesson.table(pd.DataFrame({'candidate': names, 'dot': dots, 'cosine': cosine, 'euclidean': distances}))
        display(results)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for v, name in zip(vectors, names):
            axes[0].arrow(0, 0, *v, width=.04, length_includes_head=True)
            axes[0].annotate(name, v, xytext=(3, 5), textcoords='offset points', fontsize=8)
        axes[0].set(xlim=(-1.8, 4.5), ylim=(-1, 2.5), title='Raw vectors: magnitude matters', xlabel='x', ylabel='y')
        axes[0].set_aspect('equal', adjustable='box')
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        axes[1].scatter(unit[:, 0], unit[:, 1], s=60)
        theta = np.linspace(0, 2*np.pi, 200)
        axes[1].plot(np.cos(theta), np.sin(theta), '--', color='gray')
        axes[1].set_aspect('equal')
        axes[1].set(title='Normalized vectors: only direction remains', xlabel='x', ylabel='y')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Deliberate failure
        The large-magnitude vector wins on dot product despite differing from the query direction. Neither metric is universally correct: a trained model may intentionally encode information in magnitude. The index metric must match the representation's intended use.

        **Exercise:** replace the second candidate with `[0.8, 0.1]`. Predict all three rankings, then verify. Explain why a zero vector cannot have a meaningful cosine angle.

        Use this hand-built geometry to verify formulas and indexing conventions. It does not model the semantic geometry of a language encoder, and a two-dimensional projection of a real encoder is not evidence of retrieval quality.
        """),
            code("""
        from vector_masterclass.retrieval import normalized
        zero_rejected = False
        try:
            normalized(np.array([0., 0.]))
        except ValueError:
            zero_rejected = True
        unit_dist2 = np.sum((unit - query)**2, axis=1)
        manifest = lesson.finish({'dot_favors_magnitude': int(dots.argmax()) == 1, 'cosine_favors_exact_direction': int(cosine.argmax()) == 0, 'normalized_metric_identity': bool(np.allclose(unit_dist2, 2 - 2*cosine)), 'zero_vector_rejected': zero_rejected})
        """),
        ],
    )

    notebook(
        "02",
        "Embedding behavior and limits",
        "B",
        ["01"],
        [
            md("""
        ## The engineering question
        Does a high semantic similarity score establish that two statements have the same meaning or answer the same question?

        **Learning outcomes:** inspect tokens and truncation; recognize negation and identifier failures; understand normalization and query/document roles; distinguish model configuration from application correctness.

        The default MiniLM encoder maps text to 384 dimensions and normalizes output. It has a configured 256-token limit. Unlike some asymmetric retrieval models, this model uses no special query/document prefix. The API still makes the role explicit so changing model families cannot silently omit required prefixes.
        """),
            code("""
        from vector_masterclass.models import embed, dense_model
        from vector_masterclass.config import model_config
        cfg = model_config('dense')
        texts = ['Atlas supports power over Ethernet.', 'Atlas does not support power over Ethernet.', 'Atlas error E104 means a disconnected cable.', 'Atlas error E140 means a configuration signature failure.', 'How long is the warranty?', 'The warranty lasts two years.']
        vectors = embed(texts)
        similarities = vectors @ vectors.T
        frame = lesson.table(pd.DataFrame(similarities, columns=[f'S{i}' for i in range(len(texts))]), 'similarity')
        display(pd.DataFrame({'id': [f'S{i}' for i in range(len(texts))], 'text': texts}))
        fig, ax = plt.subplots(figsize=(7, 5))
        picture = ax.imshow(similarities, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(texts)), [f'S{i}' for i in range(len(texts))])
        ax.set_yticks(range(len(texts)), [f'S{i}' for i in range(len(texts))])
        ax.set_title('Topical similarity can survive a change in factual meaning')
        fig.colorbar(picture, ax=ax, label='Cosine similarity')
        lesson.figure(fig)
        plt.show()
        """),
            code("""
        tokenizer = dense_model().tokenizer
        filler = 'Routine maintenance requires careful inspection. ' * 100
        late_fact = filler + ' The warranty lasts two years.'
        early_fact = 'The warranty lasts two years. ' + filler
        truncated_late = tokenizer(late_fact, truncation=True, max_length=cfg['max_length'])['input_ids']
        decoded = tokenizer.decode(truncated_late)
        token_table = lesson.table(pd.DataFrame([{'placement': name, 'untruncated_tokens': len(tokenizer.encode(text)), 'encoded_tokens': len(tokenizer(text, truncation=True, max_length=cfg['max_length'])['input_ids'])} for name, text in [('late', late_fact), ('early', early_fact)]]), 'truncation')
        display(token_table)
        print('Warranty present after late-fact truncation:', 'warranty' in decoded.lower())
        """),
            md("""
        ## Interpretation and exercise
        Inspect the negation pair rather than assuming a score threshold makes it safe to merge or cache answers. A high score reflects the encoder's learned geometry, not a calibrated probability that a claim is true. The truncation example is structural: content beyond the configured token limit never reaches the encoder.

        **Exercise:** move the warranty sentence earlier in the text and compare query similarity. Then replace `E104` with `E140` while keeping the surrounding words identical. Which failures require lexical matching, metadata, chunking, or a reranker?

        In production, pin tokenizer/model revisions and normalization together. Do not combine embeddings from different models merely because both have 384 dimensions. Choose a task-specific encoder and validate on your workload; use exact identifiers when the task demands them.
        """),
            code("""
        manifest = lesson.finish({'dimension_contract': vectors.shape == (6, 384), 'unit_norms': bool(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)), 'late_fact_truncated': 'warranty' not in decoded.lower(), 'explicit_role_configuration': cfg['query_prefix'] == '' and cfg['document_prefix'] == ''}, observations={'negation_cosine': float(similarities[0, 1]), 'model': cfg})
        """),
        ],
    )

    notebook(
        "03",
        "Exact and lexical retrieval baselines",
        "B",
        ["02"],
        [
            md(r"""
        ## The engineering question
        What can a simple exhaustive vector search and a lexical retriever already solve?

        **Learning outcomes:** implement cosine ranking directly; inspect BM25's exact-token behavior; establish a baseline before approximation or fusion; keep tuning and test queries separate.

        Exact search scores every eligible document. Its cost grows approximately with corpus size times embedding dimension, but for small corpora that can be simpler than building an ANN index. BM25 rewards matching query terms using inverse-document frequency and document-length normalization. It can preserve distinctions such as error codes that semantic encoders blur. Neither score scale is a relevance probability, so raw scores should not be added blindly.
        """),
            code("""
        from vector_masterclass.data import documents, queries, validate_gold
        from vector_masterclass.models import embed
        from vector_masterclass.retrieval import exact_search, LexicalIndex
        from vector_masterclass.metrics import ranking_metrics
        assert validate_gold()['human_review'] == 'approved', 'Gold labels changed; review before model comparison'
        docs = documents(tenant='public')
        ids = [d.key for d in docs]
        matrix = embed([d.text for d in docs])
        lexical = LexicalIndex([d.text for d in docs], ids)
        rows = []
        for q in queries('development'):
            if not q.answerable:
                continue
            dense = exact_search(embed([q.text], role='query')[0], matrix, ids, k=3)
            sparse = lexical.search(q.text, k=3)
            for method, hits in [('dense_exact', dense), ('BM25', sparse)]:
                metrics = ranking_metrics([id_ for id_, score in hits], q.relevance, 3)
                rows.append({'query': q.id, 'method': method, 'top_document': hits[0][0], **metrics})
        results = lesson.table(pd.DataFrame(rows))
        display(results[['query', 'method', 'top_document', 'recall', 'mrr']])
        fig, ax = plt.subplots()
        results.pivot(index='query', columns='method', values='mrr').plot.bar(ax=ax, color=['#2563eb', '#f59e0b'])
        ax.set(ylabel='Reciprocal rank at 3', ylim=(0, 1.15), title='Development queries: inspect errors before adding complexity')
        ax.legend(title='Retriever', bbox_to_anchor=(1.02, 1), loc='upper left')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Failure and interpretation
        Both retrievers return candidates even for an unsupported question. Ranking alone does not establish that the corpus contains an answer. A later lesson calibrates abstention separately. If both methods solve this tiny set, that is a reason to expand the evaluation, not proof that a production problem is solved.

        **Exercise:** inspect the Atlas E104/E140 results. Change a query to an exact identifier, then to a paraphrase. Explain the difference using term matching and semantic similarity. Do not alter the frozen test queries to make a preferred method win.

        Prefer exhaustive search when its measured latency fits a small workload. Prefer lexical lookup for exact codes or names, and measure hybrid retrieval for mixed queries. ANN is an efficiency mechanism; it cannot repair a poor embedding model.
        """),
            code("""
        unsupported = queries('development')[-1]
        assert not unsupported.answerable
        print('Unsupported query still gets lexical candidates:', lexical.search(unsupported.text, 2))
        manifest = lesson.finish({'gold_approved': validate_gold()['human_review'] == 'approved', 'aligned_vectors': matrix.shape[0] == len(ids), 'both_baselines_executed': set(results.method) == {'dense_exact', 'BM25'}, 'development_only': set(results['query']).issubset({q.id for q in queries('development')})}, observations={'baseline_means': results.groupby('method')[['recall','mrr','ndcg']].mean().to_dict()})
        """),
        ],
    )

    notebook(
        "04",
        "Evaluation before optimization",
        "T",
        ["03"],
        [
            md(r"""
        ## The engineering question
        Can an evaluation score improve even when the system has not found more useful evidence?

        **Learning outcomes:** calculate Recall@k, MRR, and nDCG; separate document relevance from chunk/evidence accounting; avoid duplicate credit; handle unanswerable and incompletely judged queries explicitly.

        Recall@k divides retrieved relevant documents by the number of relevant documents in the judgments. MRR uses the reciprocal rank of the first relevant result. nDCG discounts graded gains by rank and normalizes by the ideal ordering. We use gain $2^g-1$ and discount $\log_2(r+1)$ for one-based rank $r$.

        Our document metrics collapse repeated document revisions before taking the top k. Evidence coverage instead measures the union of retrieved character intervals inside immutable gold source spans. This keeps the evaluation target stable when chunk boundaries change. A no-answer query has no recall denominator; exclude it from relevance means and report its abstention behavior separately.
        """),
            code("""
        from vector_masterclass.metrics import ranking_metrics, aggregate, evidence_coverage
        from vector_masterclass.schema import Document, Chunk, SourceSpan, EvidenceUnit
        labels = {'A': 2, 'B': 1}
        ranked = ['noise', 'A', 'A', 'B']
        measured = ranking_metrics(ranked, labels, 3)
        display(measured)
        ideal = 3 + 1/np.log2(3)
        observed = 3/np.log2(3) + 1/2
        assert np.isclose(measured['ndcg'], observed/ideal)
        missing = ranking_metrics(['noise'], {}, 3)
        display(aggregate([measured, missing]))
        """),
            code("""
        doc = Document(id='demo', revision='1', title='Evidence', text='0123456789')
        gold = [EvidenceUnit(id='fact', alternatives=[SourceSpan(document_key=doc.key, source_hash=doc.sha256, start=0, end=10)])]
        duplicated = [Chunk.from_span(doc, 0, 4, 'small'), Chunk.from_span(doc, 2, 6, 'small')]
        complete = duplicated + [Chunk.from_span(doc, 6, 10, 'small')]
        rows = []
        for name, chunks in [('overlapping partial chunks', duplicated), ('complete source coverage', complete)]:
            result = evidence_coverage(chunks, gold)
            rows.append({'case': name, 'naive_sum_fraction': sum(c.end-c.start for c in chunks)/10, 'union_fraction': result['unit_fractions']['fact'], 'covered_at_0.8': result['coverage']})
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig, ax = plt.subplots(figsize=(9,4))
        results.set_index('case')[['naive_sum_fraction','union_fraction']].rename(columns={'naive_sum_fraction':'Naive summed length', 'union_fraction':'Union of source spans'}).plot.barh(ax=ax, color=['#f59e0b','#2563eb'])
        ax.axvline(.8, linestyle='--', color='black', label='Required evidence coverage')
        ax.set(xlabel='Fraction of gold evidence', ylabel='', title='Overlap can inflate a naive evidence metric')
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Failure, exercise, and decisions
        The partial chunks cover only 60% of the evidence even though their lengths sum to 80%. Adding more overlapping chunks could make the naive score exceed 100%. This is a metric bug, not a retrieval gain.

        **Exercise:** change the gold coverage threshold from 0.8 to 1.0 and add a one-character gap. Explain why a threshold is an evaluation convention that must be fixed before tuning. Add a second alternative source span supporting the same fact; it should not create a second required fact.

        Use supplied qrels for public-benchmark comparability, but report judgment coverage and acknowledge that unjudged does not mean known irrelevant. Use human-reviewed evidence spans for controlled chunking experiments. In production, audit query slices and user outcomes as well as aggregate scores; good ANN recall is a different question from human relevance.
        """),
            code("""
        manifest = lesson.finish({'hand_computed_ndcg': bool(np.isclose(measured['ndcg'], observed/ideal)), 'no_answer_excluded': missing['recall'] is None, 'overlap_not_double_counted': float(results.iloc[0]['union_fraction']) == .6, 'full_evidence_covered': float(results.iloc[1]['covered_at_0.8']) == 1})
        """),
        ],
    )

    notebook(
        "05a",
        "Qdrant data model and CRUD",
        "T",
        ["00", "03"],
        [
            md("""
        ## The engineering question
        What exactly is stored in a Qdrant collection, and which changes are schema changes?

        **Learning outcomes:** distinguish a collection, a point ID, named vector fields, and payload; execute create/read/update/delete; observe dimensionality validation; explain idempotent upsert.

        A point holds an ID, vector representations, and metadata payload. A named vector field fixes its representation's dimensionality and metric. Payload stores attributes such as tenant or source revision; it is not an authorization system by itself. Reusing a point ID replaces that point on upsert, while generating a new ID creates a different point.
        """),
            code("""
        from uuid import uuid4
        from qdrant_client import QdrantClient, models as qm
        from qdrant_client.http.exceptions import UnexpectedResponse
        from vector_masterclass.config import qdrant_url
        client = QdrantClient(url=qdrant_url(), timeout=10)
        name = 'vmc_05a_' + uuid4().hex[:8]
        events = []
        client.create_collection(name, vectors_config={'dense': qm.VectorParams(size=3, distance=qm.Distance.COSINE)}, sparse_vectors_config={'lexical': qm.SparseVectorParams()})
        try:
            points = [qm.PointStruct(id=1, vector={'dense':[1.,0.,0.]}, payload={'tenant':'red','revision':1}), qm.PointStruct(id=2, vector={'dense':[0.,1.,0.]}, payload={'tenant':'blue','revision':1})]
            client.upsert(name, points=points, wait=True)
            events.append({'operation':'insert two points','count':client.count(name,exact=True).count})
            client.upsert(name, points=points, wait=True)
            events.append({'operation':'repeat identical upsert','count':client.count(name,exact=True).count})
            client.set_payload(name, payload={'revision':2}, points=[1], wait=True)
            revised = client.retrieve(name, ids=[1])[0]
            dimension_rejected = False
            try:
                client.upsert(name, points=[qm.PointStruct(id=3, vector={'dense':[1.,0.]})], wait=True)
            except UnexpectedResponse as error:
                dimension_rejected = error.status_code == 400
                print('Expected dimensionality rejection:', error.status_code)
            client.delete(name, points_selector=qm.PointIdsList(points=[2]), wait=True)
            events.append({'operation':'delete point 2','count':client.count(name,exact=True).count})
            absent = client.retrieve(name, ids=[2])
        finally:
            client.delete_collection(name)
            client.close()
        results = lesson.table(pd.DataFrame(events))
        display(results)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.barh(results.operation, results['count'], color='#2563eb')
        ax.set(xlabel='Stored point count', title='Stable IDs make retries idempotent')
        ax.set_xticks([0,1,2])
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Failure and exercise
        A two-dimensional vector cannot enter a three-dimensional field. A subtler error remains possible: a different model can produce the same dimensions. The application must bind model/configuration identity to the index; server dimension checks cannot establish semantic compatibility.

        **Exercise:** repeat the upsert using fresh IDs and observe the duplicate logical records. Propose a stable ID derived from document revision and source span. Explain why the same tenant string in a payload does not stop a client from querying another tenant.

        Define vector fields when creating a collection. Adding an unconfigured sparse field through an ordinary collection update failed in our pinned-server pilot; later migrations use explicit schema contracts. Use named representations for deliberate hybrid pipelines, not for mixing unrelated spaces. Read/write acknowledgment, persistence, and background indexing are separated in lesson 05b.
        """),
            code("""
        manifest = lesson.finish({'idempotent_upsert': results['count'].tolist() == [2,2,1], 'payload_update_visible': revised.payload['revision'] == 2, 'dimension_mismatch_rejected': dimension_rejected, 'delete_visible': len(absent) == 0})
        """),
        ],
    )
    print("Authored notebooks 00–04 and 05a")


if __name__ == "__main__":
    main()
