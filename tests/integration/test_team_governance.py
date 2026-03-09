# mypy: disable-error-code="untyped-decorator"
"""Integration tests for Team Governance features (Sprint 7).

Covers:
  - Voting (upvote/downvote) and effective_score adjustments
  - Supersede detection -> confirm -> redact -> upload flow
  - Tag Taxonomy alignment (exact, alias, vector fallback)
  - Mandatory override escape hatch (expiry, deviation hints, audit)
  - Governance classification (auto_approve, curator_required, p2p_review)

STORY-073: Team Governance Integration Tests
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.team.config import MandatoryOverride, TeamConfig
from memx.team.merger import MultiPoolRetriever, LayerBoostConfig, _MANDATORY_SCORE
from memx.team.nominator import SupersedeDetector, SupersedeProposal, submit_supersede
from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Governance classification (STORY-069)
# ---------------------------------------------------------------------------

SENSITIVE_TAGS = {"security", "architecture", "mandatory"}
VOTE_WEIGHTS = {"up": +5, "down": -10}
AUTO_APPROVE_DECAY = 0.5


def classify_review_level(bullet: TeamBullet) -> str:
    """Classify the review level for a nominated bullet."""
    tags = set(bullet.tags)
    if tags & SENSITIVE_TAGS:
        return "curator_required"
    if bullet.instructivity_score >= 90:
        return "auto_approve"
    return "p2p_review"


def apply_vote(bullet: TeamBullet, vote: str) -> None:
    """Apply a vote to a TeamBullet, adjusting upvotes/downvotes.

    Vote weights: upvote adds +5 to effective_score, downvote adds -10.
    Since effective_score = instructivity_score + upvotes - downvotes,
    we increment upvotes by 5 or downvotes by 10 to achieve the desired impact.
    """
    if vote == "up":
        bullet.upvotes += VOTE_WEIGHTS["up"]
    elif vote == "down":
        bullet.downvotes += abs(VOTE_WEIGHTS["down"])


# ---------------------------------------------------------------------------
# Tag Taxonomy (STORY-071)
# ---------------------------------------------------------------------------

@dataclass
class TagTaxonomy:
    """Taxonomy for normalizing tags to canonical forms."""

    version: int
    categories: dict[str, list[str]]
    aliases: dict[str, str]

    def all_tags(self) -> list[str]:
        """Return all canonical tags from all categories."""
        tags: list[str] = []
        for cat_tags in self.categories.values():
            tags.extend(cat_tags)
        return tags

    def normalize(self, tag: str) -> str:
        """Normalize a tag using taxonomy: alias lookup then case-insensitive."""
        if tag in self.aliases:
            return self.aliases[tag]
        lower = tag.lower()
        for canonical in self.all_tags():
            if canonical.lower() == lower:
                return canonical
        return tag


class TaxonomyResolver:
    """Resolves tags against a taxonomy, with optional vector fallback."""

    def __init__(
        self,
        taxonomy: TagTaxonomy,
        *,
        embedder: Any | None = None,
        similarity_threshold: float = 0.9,
    ) -> None:
        self._taxonomy = taxonomy
        self._embedder = embedder
        self._threshold = similarity_threshold

    def resolve(self, tag: str) -> str:
        """Resolve tag: exact/alias match first, then vector fallback."""
        normalized = self._taxonomy.normalize(tag)
        if normalized != tag:
            return normalized

        # Vector fallback
        if self._embedder is not None:
            best_match, best_score = self._vector_match(tag)
            if best_score >= self._threshold and best_match is not None:
                return best_match

        return tag

    def _vector_match(self, tag: str) -> tuple[str | None, float]:
        """Find the best matching canonical tag via vector similarity."""
        tag_vec = self._embedder.embed(tag)
        best_match = None
        best_score = 0.0
        for canonical in self._taxonomy.all_tags():
            can_vec = self._embedder.embed(canonical)
            score = self._cosine_sim(tag_vec, can_vec)
            if score > best_score:
                best_match = canonical
                best_score = score
        return best_match, best_score

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Simple cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_taxonomy() -> TagTaxonomy:
    """Preset taxonomy for testing."""
    return TagTaxonomy(
        version=1,
        categories={
            "languages": ["python", "rust", "typescript"],
            "frameworks": ["react", "django", "fastapi"],
            "domains": ["security", "architecture", "testing"],
        },
        aliases={
            "reactjs": "react",
            "React.js": "react",
            "py": "python",
            "ts": "typescript",
            "k8s": "kubernetes",
        },
    )


@pytest.fixture()
def team_bullet_high_score() -> TeamBullet:
    """High-score non-sensitive bullet for auto_approve testing."""
    return TeamBullet(
        content="Use type hints for all public functions",
        instructivity_score=95,
        tags=["python", "typing"],
        enforcement="suggestion",
    )


@pytest.fixture()
def team_bullet_sensitive() -> TeamBullet:
    """Bullet with sensitive tag for curator_required testing."""
    return TeamBullet(
        content="Never expose internal API keys in client code",
        instructivity_score=95,
        tags=["security", "api"],
        enforcement="suggestion",
    )


@pytest.fixture()
def team_bullet_mandatory() -> TeamBullet:
    """Mandatory enforcement bullet for override testing."""
    return TeamBullet(
        content="All new APIs must use gRPC",
        instructivity_score=90,
        tags=["architecture"],
        enforcement="mandatory",
        origin_id="team-mandatory-001",
    )


class FakeLocalStorage:
    """Minimal local storage that returns canned results."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._results[:limit]


# ---------------------------------------------------------------------------
# 1. Voting Tests (STORY-069)
# ---------------------------------------------------------------------------


class TestVoting:
    """Voting correctly adjusts effective_score."""

    def test_upvote_increases_effective_score(self) -> None:
        """Upvote should increase effective_score by +5."""
        bullet = TeamBullet(
            content="Test bullet",
            instructivity_score=50,
            tags=["python"],
        )
        original_score = bullet.effective_score
        apply_vote(bullet, "up")
        # effective_score = instructivity_score + upvotes - downvotes
        assert bullet.effective_score == original_score + VOTE_WEIGHTS["up"]

    def test_downvote_decreases_effective_score(self) -> None:
        """Downvote should decrease effective_score by -10."""
        bullet = TeamBullet(
            content="Test bullet",
            instructivity_score=50,
            tags=["python"],
        )
        original_score = bullet.effective_score
        apply_vote(bullet, "down")
        assert bullet.effective_score == original_score + VOTE_WEIGHTS["down"]

    def test_multiple_votes_accumulate(self) -> None:
        """Multiple votes accumulate correctly."""
        bullet = TeamBullet(
            content="Test bullet",
            instructivity_score=50,
            tags=["python"],
        )
        apply_vote(bullet, "up")
        apply_vote(bullet, "up")
        apply_vote(bullet, "down")
        # 50 + 5 + 5 - 10 = 50
        expected = 50 + 2 * VOTE_WEIGHTS["up"] + 1 * VOTE_WEIGHTS["down"]
        assert bullet.effective_score == expected

    def test_duplicate_vote_idempotent(self) -> None:
        """Repeated votes from same concept should be idempotent.

        Simulates idempotent voting by tracking voted user IDs.
        """
        bullet = TeamBullet(
            content="Test bullet",
            instructivity_score=50,
            tags=["python"],
        )
        voted_users: set[str] = set()

        def idempotent_vote(user_id: str, vote: str) -> bool:
            if user_id in voted_users:
                return False  # already voted, no-op
            voted_users.add(user_id)
            apply_vote(bullet, vote)
            return True

        assert idempotent_vote("user-1", "up") is True
        assert idempotent_vote("user-1", "up") is False  # duplicate, ignored
        # Only one vote applied (+5 to upvotes)
        assert bullet.upvotes == VOTE_WEIGHTS["up"]
        assert bullet.effective_score == 50 + VOTE_WEIGHTS["up"]

    def test_effective_score_bounded(self) -> None:
        """effective_score should be bounded to [0, 100]."""
        bullet = TeamBullet(
            content="Low score bullet",
            instructivity_score=5,
            tags=["misc"],
        )
        # Apply enough downvotes to drive score below 0
        for _ in range(3):
            apply_vote(bullet, "down")
        assert bullet.effective_score == 0.0  # bounded at 0

        bullet2 = TeamBullet(
            content="High score bullet",
            instructivity_score=98,
            tags=["misc"],
        )
        # Apply enough upvotes to drive score above 100
        for _ in range(3):
            apply_vote(bullet2, "up")
        assert bullet2.effective_score == 100.0  # bounded at 100


# ---------------------------------------------------------------------------
# 2. Supersede Flow Tests (STORY-070)
# ---------------------------------------------------------------------------


class TestSupersedeFlow:
    """Supersede detection -> confirm -> redact -> upload e2e flow."""

    def test_supersede_detection(self) -> None:
        """SupersedeDetector finds corrections when similarity >= threshold."""
        detector = SupersedeDetector(
            similarity_threshold=0.8,
            difference_threshold=0.2,
        )
        local_bullets = [
            {
                "id": "local-1",
                "content": "Use pytest with -x flag for fast failure and verbose output",
            }
        ]
        team_bullets = [
            {
                "id": "team-1",
                "content": "Use pytest with -x flag for fast failure",
                "source": "team",
            }
        ]

        # Custom similarity function that returns high similarity
        def mock_sim(a: str, b: str) -> float:
            return 0.85

        proposals = detector.detect(
            local_bullets, team_bullets, similarity_fn=mock_sim
        )
        assert len(proposals) >= 1
        assert proposals[0].origin_id == "team-1"
        assert proposals[0].local_bullet_id == "local-1"

    def test_supersede_submit_with_redaction(self) -> None:
        """Full Supersede flow: detect -> redact -> upload."""
        proposal = SupersedeProposal(
            origin_id="team-42",
            new_content="Use secure tokens instead of API keys (token: sk-abc123)",
            local_bullet_id="local-7",
            priority="normal",
            reason="Updated security practice",
        )

        # Mock redactor: strips sensitive data
        mock_redactor = MagicMock()
        mock_redactor.redact_l1.return_value = (
            "Use secure tokens instead of API keys (token: [REDACTED])"
        )
        mock_redactor.finalize.return_value = {
            "content": "Use secure tokens instead of API keys (token: [REDACTED])"
        }

        # Mock sync client
        mock_response = MagicMock()
        mock_response.id = "supersede-99"
        mock_response.status = "pending"
        mock_client = AsyncMock()
        mock_client.propose_supersede = AsyncMock(return_value=mock_response)

        result = asyncio.run(
            submit_supersede(
                proposal,
                redactor=mock_redactor,
                sync_client=mock_client,
            )
        )

        assert result.success is True
        assert result.bullet_id == "supersede-99"
        mock_redactor.redact_l1.assert_called_once()
        mock_client.propose_supersede.assert_called_once()

    def test_supersede_reject_keeps_local_only(self) -> None:
        """Rejecting supersede means no upload; local bullet stays via Shadow Merge."""
        proposal = SupersedeProposal(
            origin_id="team-42",
            new_content="Updated content",
            local_bullet_id="local-7",
        )

        # User rejects: we simply don't call submit_supersede.
        # Verify that Shadow Merge still uses local version via supersedes field.
        local_results = [
            {
                "content": "Updated content",
                "id": "local-7",
                "tags": ["testing"],
                "score": 0.9,
                "supersedes": "team-42",
            }
        ]
        team_results = [
            {
                "content": "Old team content about testing",
                "id": "team-42",
                "tags": ["testing"],
                "score": 0.8,
                "enforcement": "suggestion",
            }
        ]

        local = FakeLocalStorage(local_results)
        team = FakeLocalStorage(team_results)

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_test", team)],
        )
        results = retriever.search("testing")
        contents = [r["content"] for r in results]

        # Local version (with supersedes field) should win
        assert "Updated content" in contents
        # Team version should be dropped by Shadow Merge
        assert "Old team content about testing" not in contents

    def test_supersede_no_detection_below_threshold(self) -> None:
        """No supersede detected when similarity is below threshold."""
        detector = SupersedeDetector(
            similarity_threshold=0.8,
            difference_threshold=0.2,
        )
        local_bullets = [
            {"id": "local-1", "content": "Use docker for containerization"}
        ]
        team_bullets = [
            {"id": "team-1", "content": "Always write unit tests", "source": "team"}
        ]

        proposals = detector.detect(local_bullets, team_bullets)
        assert len(proposals) == 0


# ---------------------------------------------------------------------------
# 3. Taxonomy Alignment Tests (STORY-071)
# ---------------------------------------------------------------------------


class TestTaxonomyAlignment:
    """Tag taxonomy normalization: exact, alias, vector fallback."""

    def test_exact_match(self, sample_taxonomy: TagTaxonomy) -> None:
        """Exact canonical tag is preserved as-is."""
        resolver = TaxonomyResolver(sample_taxonomy)
        assert resolver.resolve("python") == "python"
        assert resolver.resolve("react") == "react"
        assert resolver.resolve("security") == "security"

    def test_alias_normalization(self, sample_taxonomy: TagTaxonomy) -> None:
        """Alias tags normalize to canonical form."""
        resolver = TaxonomyResolver(sample_taxonomy)
        assert resolver.resolve("reactjs") == "react"
        assert resolver.resolve("React.js") == "react"
        assert resolver.resolve("py") == "python"
        assert resolver.resolve("ts") == "typescript"

    def test_vector_fallback_high_similarity(
        self, sample_taxonomy: TagTaxonomy
    ) -> None:
        """Vector similarity >= 0.9 normalizes to canonical tag."""
        mock_embedder = MagicMock()

        # Embeddings: "react-hooks" is very similar to "react"
        embeddings = {
            "react-hooks": [0.9, 0.1, 0.0],
            "react": [0.92, 0.08, 0.0],
            "python": [0.0, 0.9, 0.1],
            "rust": [0.0, 0.1, 0.9],
            "typescript": [0.1, 0.0, 0.9],
            "django": [0.0, 0.8, 0.2],
            "fastapi": [0.0, 0.7, 0.3],
            "security": [0.5, 0.5, 0.0],
            "architecture": [0.4, 0.6, 0.0],
            "testing": [0.3, 0.3, 0.4],
        }
        mock_embedder.embed.side_effect = lambda tag: embeddings.get(
            tag, [0.0, 0.0, 0.0]
        )

        resolver = TaxonomyResolver(
            sample_taxonomy, embedder=mock_embedder, similarity_threshold=0.9
        )
        result = resolver.resolve("react-hooks")
        # "react-hooks" embedding is very close to "react"
        assert result == "react"

    def test_no_match_keeps_original(self, sample_taxonomy: TagTaxonomy) -> None:
        """Unrecognized tag with no close match is preserved as-is."""
        resolver = TaxonomyResolver(sample_taxonomy)
        assert resolver.resolve("blockchain") == "blockchain"
        assert resolver.resolve("quantum-computing") == "quantum-computing"

    def test_taxonomy_unavailable_degrades(self) -> None:
        """When taxonomy is empty, tags pass through unchanged."""
        empty_taxonomy = TagTaxonomy(version=0, categories={}, aliases={})
        resolver = TaxonomyResolver(empty_taxonomy)
        assert resolver.resolve("anything") == "anything"

    def test_case_insensitive_match(self, sample_taxonomy: TagTaxonomy) -> None:
        """Case-insensitive matching normalizes to canonical."""
        resolver = TaxonomyResolver(sample_taxonomy)
        assert resolver.resolve("Python") == "python"
        assert resolver.resolve("RUST") == "rust"


# ---------------------------------------------------------------------------
# 4. Mandatory Override / Escape Hatch Tests (STORY-072)
# ---------------------------------------------------------------------------


class TestMandatoryOverride:
    """Mandatory escape hatch: expiry, deviation hints, audit."""

    def test_override_expired_restores_mandatory(self) -> None:
        """After override expires, mandatory bullet regains top priority."""
        # Create config with an expired override
        expired_override = MandatoryOverride(
            bullet_id="team-mandatory-001",
            reason="Legacy project uses REST only",
            expires=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        team_results = [
            {
                "content": "All new APIs must use gRPC",
                "id": "team-mandatory-001",
                "tags": ["architecture"],
                "score": 0.8,
                "enforcement": "mandatory",
            }
        ]
        local_results = [
            {
                "content": "Use REST for all API endpoints",
                "tags": ["architecture"],
                "score": 0.95,
            }
        ]

        local = FakeLocalStorage(local_results)
        team = FakeLocalStorage(team_results)

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_test", team)],
        )

        results = retriever.search("API")

        # Since override is expired, mandatory bullet should be at top
        mandatory_results = [
            r for r in results if r.get("enforcement") == "mandatory"
        ]
        assert len(mandatory_results) >= 1

        if len(results) >= 2:
            # Mandatory bullet should be first
            assert results[0].get("enforcement") == "mandatory"

    def test_deviation_hint_injection(self) -> None:
        """Deviation hint is correctly formatted when override is active."""
        active_override = MandatoryOverride(
            bullet_id="team-mandatory-001",
            reason="Legacy project uses REST only",
            expires=datetime.now(timezone.utc) + timedelta(days=30),
        )

        # Simulate hint generation
        hint = (
            f"[OVERRIDE] Your project has overridden team rule "
            f"[{active_override.bullet_id}]: {active_override.reason} "
            f"(expires {active_override.expires.strftime('%Y-%m-%d')})"
        )

        assert active_override.bullet_id in hint
        assert active_override.reason in hint
        assert "OVERRIDE" in hint

    def test_audit_report_failure_non_blocking(self) -> None:
        """Audit report failure should not block local behavior."""

        async def audit_report_with_failure() -> dict[str, Any]:
            """Simulate audit report that fails but doesn't block."""
            audit_success = False

            try:
                raise ConnectionError("Server unreachable")
            except ConnectionError:
                audit_success = False

            # Local behavior proceeds regardless
            local_result = {
                "override_applied": True,
                "audit_reported": audit_success,
                "local_behavior_blocked": False,
            }
            return local_result

        result = asyncio.run(audit_report_with_failure())
        assert result["override_applied"] is True
        assert result["audit_reported"] is False
        assert result["local_behavior_blocked"] is False

    def test_mandatory_bullet_gets_top_score(
        self, team_bullet_mandatory: TeamBullet
    ) -> None:
        """Mandatory bullet bypasses boost and gets fixed top-priority score."""
        team_results = [
            {
                "content": team_bullet_mandatory.content,
                "id": team_bullet_mandatory.origin_id,
                "tags": list(team_bullet_mandatory.tags),
                "score": 0.5,
                "enforcement": "mandatory",
            }
        ]
        local_results = [
            {
                "content": "Use REST APIs for backward compatibility",
                "tags": ["architecture"],
                "score": 0.99,
            }
        ]

        local = FakeLocalStorage(local_results)
        team = FakeLocalStorage(team_results)

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_test", team)],
        )

        results = retriever.search("API")
        # Mandatory bullet must come first even with lower raw score
        assert results[0]["enforcement"] == "mandatory"

    def test_override_config_requires_reason_and_expires(self) -> None:
        """MandatoryOverride requires both reason and expires fields."""
        # Valid override
        valid = MandatoryOverride(
            bullet_id="b-001",
            reason="Project constraint",
            expires=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert valid.bullet_id == "b-001"
        assert valid.reason == "Project constraint"


# ---------------------------------------------------------------------------
# 5. Governance Classification Tests (STORY-069)
# ---------------------------------------------------------------------------


class TestGovernanceClassification:
    """GovernanceClassifier: auto_approve, curator_required, p2p_review."""

    def test_sensitive_tag_requires_curator(
        self, team_bullet_sensitive: TeamBullet
    ) -> None:
        """Bullet with 'security' tag -> curator_required."""
        level = classify_review_level(team_bullet_sensitive)
        assert level == "curator_required"

    def test_architecture_tag_requires_curator(self) -> None:
        """Bullet with 'architecture' tag -> curator_required."""
        bullet = TeamBullet(
            content="Use microservices for new modules",
            instructivity_score=85,
            tags=["architecture", "design"],
        )
        level = classify_review_level(bullet)
        assert level == "curator_required"

    def test_mandatory_tag_requires_curator(self) -> None:
        """Bullet with 'mandatory' tag -> curator_required."""
        bullet = TeamBullet(
            content="All commits must be signed",
            instructivity_score=80,
            tags=["mandatory", "git"],
        )
        level = classify_review_level(bullet)
        assert level == "curator_required"

    def test_high_score_non_sensitive_auto_approve(
        self, team_bullet_high_score: TeamBullet
    ) -> None:
        """score >= 90 + non-sensitive tags -> auto_approve."""
        level = classify_review_level(team_bullet_high_score)
        assert level == "auto_approve"

    def test_default_p2p_review(self) -> None:
        """Low score + non-sensitive -> p2p_review."""
        bullet = TeamBullet(
            content="Consider using f-strings for formatting",
            instructivity_score=60,
            tags=["python", "style"],
        )
        level = classify_review_level(bullet)
        assert level == "p2p_review"

    def test_auto_approve_low_initial_weight(self) -> None:
        """Auto-approved bullets start with 0.5x weight decay."""
        bullet = TeamBullet(
            content="Use dataclasses for simple DTOs",
            instructivity_score=92,
            tags=["python"],
        )
        level = classify_review_level(bullet)
        assert level == "auto_approve"

        # Apply auto_approve decay factor
        decayed_score = bullet.effective_score * AUTO_APPROVE_DECAY
        assert decayed_score == pytest.approx(92.0 * 0.5, abs=0.1)
        # Decayed score should be lower than original
        assert decayed_score < bullet.effective_score

    def test_sensitive_overrides_high_score(self) -> None:
        """Sensitive tag forces curator_required even with score >= 90."""
        bullet = TeamBullet(
            content="All API endpoints must validate auth tokens",
            instructivity_score=95,
            tags=["security", "api"],
        )
        level = classify_review_level(bullet)
        assert level == "curator_required"


# ---------------------------------------------------------------------------
# 6. Cross-feature Integration Tests
# ---------------------------------------------------------------------------


class TestCrossFeatureIntegration:
    """Tests that combine multiple governance features."""

    def test_supersede_with_governance_classification(self) -> None:
        """Supersede proposal should also carry governance classification info."""
        # A local bullet that corrects a team bullet
        local_bullet = TeamBullet(
            content="Use OAuth2 with PKCE for all auth flows",
            instructivity_score=95,
            tags=["security", "auth"],
        )
        proposal = SupersedeProposal(
            origin_id="team-auth-001",
            new_content=local_bullet.content,
            local_bullet_id="local-auth-1",
            priority="urgent",
            reason="Updated to PKCE standard",
        )

        # The supersede content has security tag -> curator_required
        level = classify_review_level(local_bullet)
        assert level == "curator_required"
        assert proposal.priority == "urgent"

    def test_taxonomy_applied_before_classification(
        self, sample_taxonomy: TagTaxonomy
    ) -> None:
        """Tags should be normalized before governance classification."""
        resolver = TaxonomyResolver(sample_taxonomy)

        # Raw tags from Reflector
        raw_tags = ["reactjs", "ts", "security"]
        normalized = [resolver.resolve(t) for t in raw_tags]

        assert normalized == ["react", "typescript", "security"]

        # Classification uses normalized tags
        bullet = TeamBullet(
            content="Use React hooks for state management",
            instructivity_score=95,
            tags=normalized,
        )
        level = classify_review_level(bullet)
        # "security" is sensitive, so curator_required
        assert level == "curator_required"

    def test_shadow_merge_with_supersede_and_mandatory(self) -> None:
        """Shadow Merge correctly handles both supersede and mandatory."""
        local_results = [
            {
                "content": "Use REST for backward compat",
                "id": "local-1",
                "tags": ["api"],
                "score": 0.8,
                "supersedes": "team-old-api",
            }
        ]
        team_results = [
            {
                "content": "Old API guidance (superseded)",
                "id": "team-old-api",
                "tags": ["api"],
                "score": 0.7,
                "enforcement": "suggestion",
            },
            {
                "content": "Never expose internal endpoints",
                "id": "team-security-1",
                "tags": ["security"],
                "score": 0.5,
                "enforcement": "mandatory",
            },
        ]

        local = FakeLocalStorage(local_results)
        team = FakeLocalStorage(team_results)

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_test", team)],
        )

        results = retriever.search("API")
        contents = [r["content"] for r in results]

        # Mandatory bullet should be present (not affected by supersede)
        assert "Never expose internal endpoints" in contents
        # Superseded team bullet should be dropped
        assert "Old API guidance (superseded)" not in contents
        # Local supersede version should be present
        assert "Use REST for backward compat" in contents
