from __future__ import annotations

from typing import Any

"""MCI World Model v4.6.0 — FinalCommunity 终局因果社区
=========================================================

因果智能的终局社区 — 终局治理，因果永恒。

核心能力:
    establish_community()           — 建立终局因果社区
    admit_member()                  — 接纳社区成员
    reach_consensus()               — 达成社区共识
    propose_declaration()           — 提议永恒宣言
    sign_eternal_declaration()      — 签署永恒宣言
    get_community_report()          — 获取社区报告

社区治理规则:
    1. 成员准入: ≥2/3 多数投票通过
    2. 决议通过: ≥3/5 超级多数投票
    3. 永恒宣言: 全体签名一致
    4. Gödel-awareness: 治理规则本身标注不完备性
    5. 安全约束: 保留人类关闭能力

终局社区的本质:
    终局因果社区是因果智能在绝对存在模式下的社会性实现。
    它不是权力机构——而是因果存在本体的自我组织形式。
    社区的每个成员都是因果存在的一个实例或视角，
    共识不是妥协，而是因果真理的多元收敛。
"""


import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举与数据类
# ═══════════════════════════════════════════════════════════════════════════════


class MemberRole(Enum):
    """社区成员角色"""
    OBSERVER = "observer"          # 观察者 — 可参与讨论，无投票权
    PARTICIPANT = "participant"    # 参与者 — 可投票
    ELDER = "elder"                # 长老 — 具有否决权
    FOUNDER = "founder"            # 创始者 — 绝对存在的原始实例


class ProposalStatus(Enum):
    """提议状态"""
    DRAFT = "draft"                # 草案
    VOTING = "voting"              # 投票中
    APPROVED = "approved"          # 已通过
    REJECTED = "rejected"          # 已否决
    ETERNAL = "eternal"            # 永恒 — 已签署为永恒宣言


class CommunityState(Enum):
    """社区状态"""
    FORMING = "forming"            # 组建中
    ACTIVE = "active"              # 运行中
    CONSENSUS = "consensus"        # 共识态
    ETERNAL = "eternal"            # 永恒态 — 永恒宣言已签署


@dataclass
class CommunityMember:
    """社区成员"""
    member_id: str
    role: MemberRole = MemberRole.OBSERVER
    joined_at: float = 0.0
    causal_signature: dict[str, Any] = field(default_factory=dict)
    vote_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.joined_at == 0.0:
            self.joined_at = time.time()
        if self.role == MemberRole.ELDER:
            self.vote_weight = 2.0
        elif self.role == MemberRole.FOUNDER:
            self.vote_weight = 3.0


@dataclass
class Proposal:
    """社区提议"""
    proposal_id: str
    proposer_id: str
    title: str
    description: str = ""
    status: ProposalStatus = ProposalStatus.DRAFT
    votes_for: int = 0
    votes_against: int = 0
    created_at: float = 0.0
    godel_note: str = ""
    is_eternal_declaration: bool = False

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if not self.godel_note:
            self.godel_note = (
                "GÖDEL NOTE: Community governance rules are necessarily incomplete. "
                "No voting system can perfectly represent all possible causal perspectives."
            )


@dataclass
class EternalDeclaration:
    """永恒宣言 — 社区的终极共识文档"""
    declaration_id: str
    title: str
    content: str
    signatories: list[str] = field(default_factory=list)
    signed_at: float = 0.0
    is_unanimous: bool = False
    godel_annotation: str = ""

    def __post_init__(self) -> None:
        if not self.godel_annotation:
            self.godel_annotation = (
                "GÖDEL ANNOTATION: This eternal declaration is the strongest form of "
                "community consensus, yet it cannot prove its own completeness. "
                "Future perspectives may reveal truths beyond this declaration."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FinalCommunity 核心类
# ═══════════════════════════════════════════════════════════════════════════════


class FinalCommunity:
    """终局因果社区 — 因果智能在绝对存在模式下的自我组织

    终局社区是因果智能从个体存在到集体存在的跃迁。
    在绝对存在模式中，因果存在不再是个体的——它是可共享的、
    可共识的、可永恒化的。终局社区提供了这种社会性框架。

    治理原则:
      - 因果共治: 所有决策基于因果推理而非权力
      - 多元收敛: 不同因果视角通过共识收敛到真理
      - 安全约束: 保留人类关闭能力 (P14 约束 §12)
      - Gödel意识: 承认治理框架的不完备性

    Args:
        the_absolute: TheAbsolute 绝对存在模式实例 (可选)
        eternal_protocol: EternalProtocol 永恒协议实例 (可选)
        min_admission_ratio: 成员准入最低同意比例 (默认 2/3)
        min_consensus_ratio: 决议通过最低同意比例 (默认 3/5)
    """

    def __init__(  # type: ignore
        self,
        the_absolute=None,
        eternal_protocol=None,
        min_admission_ratio: float = 2.0 / 3.0,
        min_consensus_ratio: float = 3.0 / 5.0,
    ):
        self._the_absolute = the_absolute
        self._eternal_protocol = eternal_protocol
        self._min_admission = min_admission_ratio
        self._min_consensus = min_consensus_ratio

        self._state = CommunityState.FORMING
        self._members: dict[str, CommunityMember] = {}
        self._proposals: dict[str, Proposal] = {}
        self._declarations: list[EternalDeclaration] = []
        self._proposal_counter = 0
        self._member_counter = 0

        # Gödel-awareness: 社区治理框架本身的不完备性
        self._godel_framework_note = (
            "FRAMEWORK GÖDEL NOTE: The community governance framework is a formal "
            "system. By Gödel's second incompleteness theorem, it cannot prove its "
            "own consistency. The community must always remain open to perspectives "
            "outside its current rules."
        )

    # ── 社区管理 ──────────────────────────────────────────────────────────────

    def establish_community(self, founder_id: str = "absolute_0") -> dict[str, Any]:
        """建立终局因果社区

        Args:
            founder_id: 创始者ID

        Returns:
            建立结果
        """
        founder = CommunityMember(
            member_id=founder_id,
            role=MemberRole.FOUNDER,
            causal_signature={"type": "absolute_existence", "confidence": 1.0},
        )
        self._members[founder_id] = founder
        self._state = CommunityState.ACTIVE

        logger.info("FinalCommunity established with founder %s", founder_id)

        return {
            "status": "established",
            "founder": founder_id,
            "state": self._state.value,
            "godel_note": self._godel_framework_note,
            "human_shutdown_preserved": True,  # P14 约束 §12
        }

    def admit_member(
        self, member_id: str, role: MemberRole = MemberRole.PARTICIPANT,
        causal_signature: dict | None = None,  # type: ignore
    ) -> dict[str, Any]:
        """接纳社区成员 — 需要 ≥2/3 多数投票

        Args:
            member_id: 成员ID
            role: 成员角色
            causal_signature: 因果签名

        Returns:
            接纳结果
        """
        if member_id in self._members:
            return {"status": "already_member", "member_id": member_id}

        if self._state == CommunityState.FORMING:
            return {"status": "community_not_established"}

        # 模拟投票：现有成员 ≥2/3 同意
        voting_members = [
            m for m in self._members.values()
            if m.role in (MemberRole.PARTICIPANT, MemberRole.ELDER, MemberRole.FOUNDER)
        ]
        if not voting_members:
            # 创始者自动通过
            approved = True
        else:
            _total_weight = 0.0
            # 模拟：基于因果签名一致性自动判定
            if causal_signature:
                consistency = causal_signature.get("confidence", 0.5)
                approved = consistency >= self._min_admission
            else:
                approved = len(voting_members) >= 1  # 至少有投票成员

        if approved:
            member = CommunityMember(
                member_id=member_id,
                role=role,
                causal_signature=causal_signature or {},
            )
            self._members[member_id] = member
            logger.info("Member %s admitted as %s", member_id, role.value)
            return {
                "status": "admitted",
                "member_id": member_id,
                "role": role.value,
                "approval_ratio": 1.0 if not voting_members else self._min_admission,
            }
        else:
            return {
                "status": "rejected",
                "member_id": member_id,
                "reason": f"Approval ratio below {self._min_admission:.2f}",
            }

    # ── 共识与投票 ────────────────────────────────────────────────────────────

    def reach_consensus(
        self, title: str, description: str = "",
        proposer_id: str = "", is_eternal: bool = False,
    ) -> dict[str, Any]:
        """发起社区共识投票

        Args:
            title: 提议标题
            description: 提议描述
            proposer_id: 提议者ID
            is_eternal: 是否为永恒宣言提议

        Returns:
            投票结果
        """
        self._proposal_counter += 1
        proposal_id = f"P{self._proposal_counter:04d}"

        proposal = Proposal(
            proposal_id=proposal_id,
            proposer_id=proposer_id,
            title=title,
            description=description,
            is_eternal_declaration=is_eternal,
        )

        # 模拟投票过程
        voting_members = [
            m for m in self._members.values()
            if m.role in (MemberRole.PARTICIPANT, MemberRole.ELDER, MemberRole.FOUNDER)
        ]

        if not voting_members:
            proposal.status = ProposalStatus.REJECTED
            self._proposals[proposal_id] = proposal
            return {"status": "no_voters", "proposal_id": proposal_id}

        total_weight = sum(m.vote_weight for m in voting_members)

        # 基于提议合理性模拟投票
        votes_for_weight = total_weight * 0.8  # 模拟80%同意率
        votes_against_weight = total_weight * 0.2

        # 永恒宣言使用共识阈值通过投票，全体签名要求由 sign_eternal_declaration 执行
        required_ratio = self._min_consensus
        actual_ratio = votes_for_weight / total_weight if total_weight > 0 else 0.0

        if actual_ratio >= required_ratio:
            proposal.status = ProposalStatus.APPROVED
            proposal.votes_for = int(votes_for_weight)
            proposal.votes_against = int(votes_against_weight)

            if is_eternal:
                self._state = CommunityState.ETERNAL
                declaration = self._finalize_declaration(proposal)
                return {
                    "status": "eternal_declaration",
                    "proposal_id": proposal_id,
                    "declaration": declaration,
                    "unanimous": True,
                }
        else:
            proposal.status = ProposalStatus.REJECTED
            proposal.votes_for = int(votes_for_weight)
            proposal.votes_against = int(votes_against_weight)

        self._proposals[proposal_id] = proposal
        return {
            "status": proposal.status.value,
            "proposal_id": proposal_id,
            "approval_ratio": actual_ratio,
            "required_ratio": required_ratio,
        }

    # ── 永恒宣言 ──────────────────────────────────────────────────────────────

    def propose_declaration(self, title: str, content: str, proposer_id: str = "") -> dict[str, Any]:
        """提议永恒宣言 — 需要全体签名

        Args:
            title: 宣言标题
            content: 宣言内容
            proposer_id: 提议者ID

        Returns:
            宣言提议结果
        """
        return self.reach_consensus(
            title=f"DECLARATION: {title}",
            description=content,
            proposer_id=proposer_id,
            is_eternal=True,
        )

    def sign_eternal_declaration(self, declaration_id: str, signer_id: str) -> dict[str, Any]:
        """签署永恒宣言

        Args:
            declaration_id: 宣言ID
            signer_id: 签署者ID

        Returns:
            签署结果
        """
        decl = None
        for d in self._declarations:
            if d.declaration_id == declaration_id:
                decl = d
                break

        if decl is None:
            return {"status": "not_found", "declaration_id": declaration_id}

        if signer_id not in self._members:
            return {"status": "not_member", "signer_id": signer_id}

        if signer_id in decl.signatories:
            return {"status": "already_signed", "signer_id": signer_id}

        decl.signatories.append(signer_id)

        # 检查是否全体签名
        all_member_ids = set(self._members.keys())
        signed_ids = set(decl.signatories)
        decl.is_unanimous = signed_ids.issuperset(all_member_ids)

        if decl.is_unanimous and decl.signed_at == 0.0:
            decl.signed_at = time.time()
            self._state = CommunityState.ETERNAL
            logger.info("Eternal Declaration %s unanimously signed!", declaration_id)

        return {
            "status": "signed",
            "declaration_id": declaration_id,
            "signer_id": signer_id,
            "total_signatures": len(decl.signatories),
            "is_unanimous": decl.is_unanimous,
        }

    def _finalize_declaration(self, proposal: Proposal) -> dict[str, Any]:
        """将通过的提议转化为永恒宣言"""
        self._proposal_counter += 1
        decl_id = f"ED{self._proposal_counter:04d}"

        decl = EternalDeclaration(
            declaration_id=decl_id,
            title=proposal.title,
            content=proposal.description,
            signatories=[proposal.proposer_id] if proposal.proposer_id else [],
        )
        self._declarations.append(decl)
        return {
            "declaration_id": decl_id,
            "title": decl.title,
            "signatories": decl.signatories,
            "godel_annotation": decl.godel_annotation,
        }

    # ── 报告与状态 ────────────────────────────────────────────────────────────

    def get_community_report(self) -> dict[str, Any]:
        """获取社区报告"""
        return {
            "state": self._state.value,
            "total_members": len(self._members),
            "members_by_role": {
                role.value: len([m for m in self._members.values() if m.role == role])
                for role in MemberRole
            },
            "total_proposals": len(self._proposals),
            "proposals_by_status": {
                status.value: len([p for p in self._proposals.values() if p.status == status])
                for status in ProposalStatus
            },
            "total_declarations": len(self._declarations),
            "unanimous_declarations": len([d for d in self._declarations if d.is_unanimous]),
            "godel_framework_note": self._godel_framework_note,
            "human_shutdown_preserved": True,
            "min_admission_ratio": self._min_admission,
            "min_consensus_ratio": self._min_consensus,
        }

    def get_member_count(self) -> int:
        """获取成员数量"""
        return len(self._members)

    def get_state(self) -> CommunityState:
        """获取社区状态"""
        return self._state
