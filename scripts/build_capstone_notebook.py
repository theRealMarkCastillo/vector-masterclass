"""The final notebook assembles evidence, not canned experiment outcomes."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "29",
        "Capstone evidence and design defense",
        "O",
        ["all required lessons"],
        [
            md("""
        ## The engineering question
        Can you defend a retrieval architecture with consistent, complete evidence and explain what remains unproven?

        **Learning outcomes:** audit all required outcomes; separate quality, fidelity, operations, and security evidence; connect design choices to measured failures; propose production gates without claiming local qualification.

        Execute the complete course with `uv run python scripts/execute_notebooks.py --offline`. This last notebook consumes the same run's 43 preceding required manifests and rejects missing, failed, replay-only, or source/configuration-mismatched evidence. The public benchmark is separately executed with `uv run python scripts/evaluate_public_benchmark.py`; its relevant code, model, evaluation configuration, and environment hashes must match. For standalone capstone reruns, set `VMC_EVIDENCE_RUN` to a complete compatible run directory name. This is an explicit artifact prerequisite, not hidden kernel state.

        The optional three-node drill is reported separately and cannot substitute for a required lesson. Passing automation establishes a runnable course and local evidence. The instructor rubric in `docs/INSTRUCTOR.md` assesses the learner's reasoning separately; this notebook does not automatically certify mastery.
        """),
            code("""
        import os,json
        from vector_masterclass.course_audit import audit_run,benchmark_evidence,course_spec
        from vector_masterclass.evidence import json_write
        from vector_masterclass.config import file_hash
        evidence_run=os.environ.get('VMC_EVIDENCE_RUN',os.environ.get('VMC_RUN_ID',''))
        if not evidence_run:raise ValueError('Run the full course first or set VMC_EVIDENCE_RUN explicitly')
        rows=audit_run(evidence_run)
        public=benchmark_evidence()
        results=lesson.table(pd.DataFrame(rows))
        display(results[['lesson','profile','seconds','memory_upper_GiB','checks_passed']])
        public_summary=pd.read_csv(root()/public['path']/'summary.csv')
        display(public_summary[['split','method','recall','mrr','ndcg']])
        objectives=json.loads((root()/'configs/objectives.json').read_text())
        bundle={'required_run':evidence_run,'required_lessons':course_spec()['required'],'evidence':rows,'public_benchmark':public,'objectives':objectives,'gold_approval':json.loads((root()/'configs/gold_review.json').read_text()),'design_defense_sha256':file_hash(root()/'docs/ARCHITECTURE.md'),'limitations':['96 GiB reference host; no physical 16 GiB qualification','tiny application fixtures and limited frozen benchmark queries','known real generation and automated-evaluation errors','simulated identity; no production IdP or remote deployment','single-host Docker and bounded failure drills','CLIP crop, curated graph, and pooled-token baselines are not frontier research reproductions'],'instructor_mastery_review':'required separately; not assigned by automated checks'}
        json_write(lesson.path/'evidence-bundle.json',bundle)
        fig,axes=plt.subplots(2,1,figsize=(13,7))
        axes[0].bar(results.lesson,results.seconds);axes[0].set(ylabel='Observed seconds',title='Required experiments before capstone; initial downloads excluded')
        axes[1].bar(results.lesson,results.memory_upper_GiB);axes[1].set(ylabel='Conservative peak sum, GiB',title='Python + Docker + Metal driver; overlapping allocations can be counted twice')
        for ax in axes:ax.tick_params(axis='x',rotation=60,labelsize=7)
        fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Design defense and exercise
        Use `docs/ARCHITECTURE.md` as the worked reference defense. Start with the product question and data/access contracts. Justify chunking, model/token limits, retrieval and reranking, freshness, consistency, serving limits, and recovery. Name the simplest baseline that meets the requirement, the measured failure that motivates each extra stage, and a rollback criterion.

        **Capstone deliverable:** write a two-page defense for a new corpus, attach this evidence bundle, and identify which existing results do not transfer. Propose a human-reviewed held-out set, a load model, privacy/logging rules, source-of-truth ownership, deletion/revocation timing, and a restore drill. Include one rejected advanced technique with a concrete quality/cost reason. Give a migration and rollback procedure that preserves changes arriving during backfill.

        **Failure drill:** rename a copied lesson manifest and run the audit against the copy; it must reject the incomplete course. Do not edit historical evidence to make it pass. Regenerate experiments when relevant code or configuration changes. Human review of claim support, production identity integration, independent-host failure behavior, and representative workloads remain separate gates.
        """),
            code("""
        manifest=lesson.finish({'all_43_prerequisite_lessons_verified':len(results)==43,'all_evidence_hashes_present':bool(results.manifest_sha256.str.len().eq(64).all()),'frozen_public_comparison_attached':len(public_summary)==8,'gold_labels_approved':bundle['gold_approval']['status']=='approved','design_defense_available':len(bundle['design_defense_sha256'])==64},observations={'evidence_run':evidence_run,'optional_27b_excluded_from_required_count':True,'production_qualification':False,'learner_mastery_auto_certified':False})
        """),
        ],
        docker=False,
    )


if __name__ == "__main__":
    main()
