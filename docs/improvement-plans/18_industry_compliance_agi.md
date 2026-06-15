# Ch18 行业落地、合规与AGI集成 — 改进规划书

## 1. 章节概述

本章节覆盖 P7-P8 波次中**行业落地、监管合规与 AGI 集成**的规划，填补原 Ch07 (经济成本)、Ch09 (替代性评估)、Ch12 (统一路径)、Ch14 (战略结论) 在以下方向的空白：

- **行业 SDK 体系**: 医疗/法律/工程 3 领域因果 SDK — 原系统无行业 SDK
- **可审计因果推理**: 100% 审计轨迹 + 报告导出 — 原系统推理不可审计
- **合规规则引擎**: 医疗/法律/工程 3 领域法规自动检查 — 原系统无合规能力
- **边缘云混合部署**: 低延迟边缘 + 复杂推理云端 — 原系统仅云端部署
- **自动伸缩**: QPS 驱动推理服务伸缩 — 原系统无自动伸缩
- **插件体系**: 第三方领域插件接口 — 原系统无插件体系
- **AGI 集成协议**: 5 端点标准接口 — 原系统无标准 AGI 接口

> **新增定位**: Ch07 覆盖"经济成本分析 (TCO)"，Ch09 覆盖"替代性目标评估"，Ch18 覆盖"行业落地、合规与 AGI 集成"——是从成本评估/替代性分析到真正行业落地和 AGI 协议级集成的关键跃迁。

## 2. 改进目标

| # | 目标 | 量化指标 | 波次 | 优先级 |
|---|---|---|---|---|
| G1 | MedicalCausalSDK | 医疗因果链追溯 20 例准确率 ≥85% | P7 | 中 |
| G2 | LegalComplianceSDK | 法律因果分析 15 例准确率 ≥80% | P7 | 中 |
| G3 | EngineeringSafetySDK | 安全推理准确率 ≥80% | P7 | 中 |
| G4 | MCIDomainSDK 统一接口 | 3 领域 SDK 统一入口 | P7 | 中 |
| G5 | AuditableCausalReasoning | 100% 审计轨迹 + 报告可导出 | P7 | 中 |
| G6 | ComplianceRuleEngine | 3 领域法规自动检查 | P7 | 中 |
| G7 | EdgeCloudHybrid | 边缘 <10ms，云端兜底 <200ms | P7 | 中 |
| G8 | AutoScaler | 100→1000 QPS 伸缩 <5s | P7 | 中 |
| G9 | MCIPluginInterface | 插件 API 规范 + ≥2 示例插件 | P7 | 中 |
| G10 | AGIIntegrationProtocol | 5 端点标准接口 + REST/gRPC | P8 | 中 |

## 3. 实施方案

### 3.1 行业 SDK 体系 (G1-G4)

**缺口**: 原系统无行业 SDK，无法与医疗/法律/工程 IT 系统集成

#### 3.1.1 MedicalCausalSDK (G1)

```python
class MedicalCausalSDK:
    """医疗因果推理 SDK — 面向医疗IT系统集成"""
    def __init__(self, world_model):
        self._wm = world_model
        self._domain_knowledge = MedicalDomainKnowledge()
    
    def diagnose_causal_chain(self, symptoms: list[str], patient_data: dict) -> dict:
        """从症状追溯因果链"""
        causal_graph = self._domain_knowledge.get_medical_graph()
        chains = self._wm.trace_causal_chain(
            source_nodes=symptoms,
            graph=causal_graph,
            max_depth=5
        )
        return {
            "symptoms": symptoms,
            "causal_chains": chains,
            "confidence": self._compute_confidence(chains),
            "treatment_implications": self._suggest_treatments(chains),
        }
    
    def predict_drug_interaction(self, drug_a: str, drug_b: str, patient: dict) -> dict:
        """预测药物交互作用"""
        interaction_graph = self._domain_knowledge.get_drug_graph()
        result = self._wm.predict_effect(
            cause=f"{drug_a}+{drug_b}",
            context=patient,
            graph=interaction_graph
        )
        return {
            "drug_pair": (drug_a, drug_b),
            "interaction_risk": result.risk_level,
            "mechanism": result.explanation,
            "recommendation": result.safety_advice,
        }
    
    def generate_safety_report(self, query: str, result: dict) -> dict:
        """生成可审计安全报告"""
        return {
            "query": query,
            "result": result,
            "causal_proof": self._wm.export_causal_proof(result),
            "safety_checks": self._wm.run_safety_checks(result),
            "timestamp": time.time(),
            "sdk_version": "1.0.0",
        }
```

**文件**: `_medical_causal_sdk.py` (~400 行)

#### 3.1.2 LegalComplianceSDK (G2)

```python
class LegalComplianceSDK:
    """法律合规 SDK — 因果推理+法规自动检查"""
    def __init__(self, world_model):
        self._wm = world_model
        self._legal_knowledge = LegalDomainKnowledge()
    
    def analyze_liability(self, action: str, outcome: str, context: dict) -> dict:
        """法律责任因果分析"""
        liability_graph = self._legal_knowledge.get_liability_graph()
        result = self._wm.trace_causal_chain(
            source_nodes=[action],
            graph=liability_graph,
            max_depth=3
        )
        return {
            "action": action,
            "outcome": outcome,
            "causal_responsibility": result.responsibility_score,
            "legal_basis": result.legal_references,
            "compliance_status": self._check_compliance(result),
        }
```

**文件**: `_legal_compliance_sdk.py` (~350 行)

#### 3.1.3 EngineeringSafetySDK (G3)

```python
class EngineeringSafetySDK:
    """工程安全 SDK — 结构安全因果推理"""
    def __init__(self, world_model):
        self._wm = world_model
        self._engineering_knowledge = EngineeringDomainKnowledge()
    
    def assess_causal_risk(self, design_params: dict, environment: dict) -> dict:
        """评估工程设计中的因果风险"""
        risk_graph = self._engineering_knowledge.get_risk_graph()
        result = self._wm.predict_effect(
            cause=design_params,
            context=environment,
            graph=risk_graph
        )
        return {
            "risk_level": result.risk_level,
            "causal_factors": result.contributing_factors,
            "mitigation_suggestions": result.safety_recommendations,
        }
```

**文件**: `_engineering_safety_sdk.py` (~300 行)

#### 3.1.4 MCIDomainSDK 统一接口 (G4)

```python
class MCIDomainSDK:
    """MCI 行业 SDK 统一入口"""
    def __init__(self, domain: str = "general"):
        self._domain = domain
        self._sdk_map = {
            "medical": MedicalCausalSDK,
            "legal": LegalComplianceSDK,
            "engineering": EngineeringSafetySDK,
        }
    
    def create_sdk(self, domain: str, **kwargs):
        """工厂方法: 创建领域 SDK"""
        sdk_cls = self._sdk_map.get(domain)
        if not sdk_cls:
            raise ValueError(f"Unsupported domain: {domain}")
        return sdk_cls(**kwargs)
    
    def list_domains(self) -> list[str]:
        return list(self._sdk_map.keys())
```

**文件**: `_domain_sdk_base.py` (~200 行)

### 3.2 监管合规框架 (G5-G6)

**缺口**: 原系统推理过程不可审计，无合规检查能力

#### 3.2.1 AuditableCausalReasoning (G5)

```python
class AuditableCausalReasoning:
    """可审计因果推理 — 满足监管合规要求"""
    def __init__(self, causal_engine, audit_logger):
        self._engine = causal_engine
        self._logger = audit_logger
        self._audit_trail: list[dict] = []
    
    def reason_with_audit(self, query: str, context: dict) -> dict:
        """带审计轨迹的因果推理"""
        # 记录推理请求
        request_id = self._logger.log_request(query, context)
        
        # 逐步推理，每步记录
        steps = []
        step1 = self._engine.parse_query(query)
        steps.append({"step": "parse", "detail": step1, "timestamp": time.time()})
        
        step2 = self._engine.identify_confounders(step1)
        steps.append({"step": "confounders", "detail": step2, "timestamp": time.time()})
        
        step3 = self._engine.estimate_effect(step1, step2, context)
        steps.append({"step": "estimate", "detail": step3, "timestamp": time.time()})
        
        step4 = self._engine.verify_safety(step3)
        steps.append({"step": "safety", "detail": step4, "timestamp": time.time()})
        
        result = {"query": query, "answer": step3, "safety": step4, "audit_steps": steps}
        
        # 记录推理结果
        self._audit_trail.append({
            "request_id": request_id,
            "steps": steps,
            "result_hash": hash(str(result)),
        })
        
        return result
    
    def export_audit_report(self, request_id: str) -> dict:
        """导出审计报告 (满足监管要求)"""
        trail = [t for t in self._audit_trail if t["request_id"] == request_id]
        return {
            "request_id": request_id,
            "reasoning_chain": trail,
            "compliance_checks": self._run_compliance(trail),
            "certification": self._generate_certificate(trail),
        }
```

**文件**: `_auditable_causal.py` (~350 行)

**审计轨迹四步模型**: parse → confounders → estimate → safety

#### 3.2.2 ComplianceRuleEngine (G6)

```python
class ComplianceRuleEngine:
    """合规规则引擎 — 行业法规自动检查"""
    def __init__(self):
        self._rule_sets = {
            "medical": MedicalComplianceRules(),
            "legal": LegalComplianceRules(),
            "engineering": EngineeringComplianceRules(),
        }
    
    def check_compliance(self, domain: str, reasoning_result: dict) -> dict:
        """检查推理结果是否符合行业法规"""
        rules = self._rule_sets.get(domain)
        if not rules:
            return {"compliant": False, "error": f"Unknown domain: {domain}"}
        
        violations = rules.check(reasoning_result)
        return {
            "domain": domain,
            "compliant": len(violations) == 0,
            "violations": violations,
            "certification": rules.generate_cert(reasoning_result) if not violations else None,
        }
```

**文件**: `_compliance_engine.py` (~250 行)

**3 领域规则集**:

| 领域 | 规则数 | 典型规则 |
|---|---|---|
| 医疗 | ≥20 | 药物交互禁忌、诊断置信度阈值、知情同意检查 |
| 法律 | ≥15 | 因果责任阈值、证据链完整性、时效性检查 |
| 工程 | ≥10 | 安全系数阈值、冗余设计要求、环境条件检查 |

### 3.3 部署与生态 (G7-G9)

#### 3.3.1 EdgeCloudHybrid (G7)

```python
class EdgeCloudHybridDeployer:
    """边缘-云混合部署架构"""
    def __init__(self, edge_config: dict, cloud_config: dict):
        self._edge = EdgeNode(edge_config)   # 轻量推理
        self._cloud = CloudNode(cloud_config)  # 重量训练/蒸馏
    
    def route_request(self, query: dict) -> str:
        """请求路由: 低延迟走边缘, 复杂推理走云端"""
        complexity = self._estimate_complexity(query)
        latency_budget = query.get("latency_budget_ms", 100)
        
        if complexity < 0.4 and latency_budget < 20:
            return "edge"   # 简单+低延迟 → 边缘
        elif complexity > 0.7:
            return "cloud"  # 复杂 → 云端
        else:
            return "edge_with_cloud_fallback"  # 边缘优先+云端兜底
    
    def sync_models(self):
        """模型同步: 云端训练 → 边缘部署"""
        cloud_model = self._cloud.get_latest_model()
        compressed = self._compress_model(cloud_model, target="edge")
        self._edge.deploy_model(compressed)
```

**文件**: `_edge_cloud_hybrid.py` (~300 行)

#### 3.3.2 AutoScaler (G8)

```python
class AutoScaler:
    """推理服务自动伸缩"""
    def __init__(self, min_replicas=1, max_replicas=10, target_latency_ms=50):
        self._min = min_replicas
        self._max = max_replicas
        self._target = target_latency_ms
    
    def compute_desired_replicas(self, current_qps: float, avg_latency_ms: float) -> int:
        """根据 QPS 和延迟计算目标副本数"""
        if avg_latency_ms > self._target * 1.5:
            scale_up = int(current_qps / 100) + 1
            return min(scale_up, self._max)
        elif avg_latency_ms < self._target * 0.5:
            scale_down = max(int(current_qps / 150), self._min)
            return scale_down
        return max(int(current_qps / 100), self._min)
```

**文件**: `_auto_scaler.py` (~150 行)

#### 3.3.3 MCIPluginInterface (G9)

```python
class MCIPluginInterface:
    """MCI 插件 API 规范"""
    name: str
    version: str
    domain: str  # medical / legal / engineering / custom
    
    def on_load(self, world_model):
        """插件加载时调用"""
        pass
    
    def on_query(self, query: str, context: dict) -> dict:
        """查询拦截/增强"""
        return {"action": "pass"}  # pass=不干预
    
    def on_result(self, query: str, result: dict) -> dict:
        """结果后处理"""
        return result
```

**文件**: 插件 API 规范文档 + `_plugin_interface.py` (~100 行)

**示例插件**:
1. `MCIPluginMedicalZhCN`: 中文医疗术语本地化插件
2. `MCIPluginSafetyOverride`: 安全约束覆盖插件 (用于特殊领域)

### 3.4 AGI 集成协议 (G10)

**缺口**: 原系统无标准 AGI 接口，LLM 无法协议级调用因果增强

```python
class AGIIntegrationProtocol:
    """AGI 集成协议 — MCI 作为因果增强层的标准接口"""
    
    PROTOCOL_VERSION = "1.0.0"
    
    # 协议定义: LLM/AGI 系统如何调用 MCI 因果增强
    ENDPOINTS = {
        "causal_query": "/v1/causal/query",          # 因果查询
        "counterfactual": "/v1/causal/counterfactual", # 反事实推理
        "safety_check": "/v1/safety/check",           # 安全约束检查
        "intervention": "/v1/causal/intervention",     # 干预分析
        "explain": "/v1/causal/explain",              # 因果解释
    }
    
    def handle_request(self, endpoint: str, payload: dict) -> dict:
        """处理 AGI 请求"""
        if endpoint not in self.ENDPOINTS:
            return {"error": f"Unknown endpoint: {endpoint}"}
        
        # 标准请求处理流程
        validated = self._validate_request(payload)
        if not validated["valid"]:
            return {"error": validated["message"]}
        
        # 调用 MCI 因果引擎
        result = self._dispatch(endpoint, payload)
        
        # 包装标准响应
        return {
            "protocol_version": self.PROTOCOL_VERSION,
            "endpoint": endpoint,
            "result": result,
            "causal_proof": self._extract_proof(result),
            "audit_trail": self._generate_audit_trail(endpoint, payload, result),
        }
    
    def _validate_request(self, payload: dict) -> dict:
        """请求验证"""
        required = ["query", "context"]
        for field in required:
            if field not in payload:
                return {"valid": False, "message": f"Missing field: {field}"}
        return {"valid": True}
```

**文件**: `_agi_protocol.py` (~350 行) + AGI 协议规范文档 (~300 行)

**协议端点详细说明**:

| 端点 | 方法 | 功能 | 输入 | 输出 |
|---|---|---|---|---|
| `/v1/causal/query` | POST | 因果查询 | query, context | ate, confidence, proof |
| `/v1/causal/counterfactual` | POST | 反事实推理 | factual, intervention | counterfactual outcome |
| `/v1/safety/check` | POST | 安全约束检查 | action, context | safe/unsafe, constraints |
| `/v1/causal/intervention` | POST | 干预分析 | do(X=x), outcome | causal effect |
| `/v1/causal/explain` | POST | 因果解释 | query, result | explanation, audit |

**传输协议**: REST (JSON) + gRPC (Protocol Buffers)

## 4. 时间计划

| 周 | 任务 | 交付物 | 波次 |
|---|---|---|---|
| W71-73 | MedicalCausalSDK + AuditableCausalReasoning | SDK核心 + 审计框架 | P7 |
| W74-76 | MedicalSDK 完善 + ComplianceRuleEngine + ScientificDiscoveryPipeline | 合规引擎 + 发现管线 | P7 |
| W77-80 | LegalComplianceSDK + ComplianceRuleEngine 完善 + HypothesisGenerator | 法律SDK + 3领域规则 | P7 |
| W81-82 | EngineeringSafetySDK + MCIDomainSDK 统一接口 | 工程SDK + 统一入口 | P7 |
| W83-86 | EdgeCloudHybrid + 开源社区 + MCIPluginInterface | 边缘云 + 社区 + 插件 | P7 |
| W87-88 | AutoScaler + 合作伙伴 | 伸缩 + 伙伴计划 | P7 |
| W89-90 | v7.0.0 发布 | 版本发布 | P7 |
| W91-93 | AGIIntegrationProtocol 设计 | 协议规范 | P8 |
| W94-96 | AGI 协议 REST/gRPC 实现 | API 服务 | P8 |
| W97-102 | 因果增强层协议规范定稿 | 终版协议文档 | P8 |

## 5. 资源配置

| 资源 | 角色 | 人天 | 说明 |
|---|---|---|---|
| 工程师 A (P7) | 行业 SDK + 边缘云 | 20 | P7 核心 |
| 工程师 B (P7) | 可审计推理 + 合规 + 社区 | 15 | P7 合规 |
| Tech Lead (P7) | 插件体系 + 战略 + 发布 | 10 | P7 生态 |
| 研究工程师 (P7) | 科学发现管线 | 15 | P7 Ch11 |
| 工程师 (P8) | AGI 协议实现 + WMMM | 10 | P8 协议 |
| Tech Lead (P8) | 协议设计 + 战略 + 发布 | 15 | P8 终局 |
| **合计** | | **85** | |

## 6. KPI 指标

| KPI | 基线 | P7 目标 | P8 目标 |
|---|---|---|---|
| 医疗SDK因果链准确率 | ≥80% (P4) | ≥85% | — |
| 法律SDK因果分析准确率 | ≥75% (P4) | ≥80% | — |
| 工程SDK安全推理准确率 | N/A | ≥80% | — |
| SDK 统一接口 | 0 | 3 领域 | — |
| 审计轨迹覆盖率 | 0% | 100% | — |
| 合规规则领域 | 0 | 3 | — |
| 边缘推理延迟 | <50ms (P3) | <10ms | — |
| 自动伸缩 | N/A | 100→1000 QPS <5s | — |
| 插件 API | N/A | 规范 + ≥2 示例 | — |
| AGI 协议端点 | 0 | — | 5 |
| 协议标准化 | N/A | — | 100% |

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 | 应急方案 |
|---|---|---|---|---|
| 医疗 SDK 审核不通过 | 中 | 高 | 增加领域专家审核 | 标注为"研究原型" |
| 合规框架覆盖不足 | 高 | 中 | 迭代增加规则 | 优先覆盖核心法规 |
| 边缘云同步延迟过高 | 中 | 中 | 增量同步 + 模型压缩 | 全云端部署 |
| 开源社区反响冷淡 | 中 | 中 | 主动推广 + 技术博客 | 降低社区目标 |
| AGI 协议与 LLM 不兼容 | 中 | 高 | 参考 OpenAI/Anthropic API 规范 | 仅做 REST 接口 |
| 20 周 (P7) 时间不够 | 中 | 高 | 工程SDK可推迟到P8 | 压缩社区建设周期 |

## 8. 成本预算

| 项目 | 人天 | 硬件/软件 | 波次 |
|---|---|---|---|
| 医疗因果 SDK | 8 | $300 (专家) | P7 |
| 法律合规 SDK | 6 | $500 (法律顾问) | P7 |
| 工程安全 SDK | 4 | $0 | P7 |
| 可审计推理框架 | 8 | $0 | P7 |
| 合规规则引擎 | 4 | $0 | P7 |
| 边缘云混合 | 5 | $200 (设备) | P7 |
| 自动伸缩 | 3 | $0 | P7 |
| 开源社区+插件 | 5 | $500 (运营) | P7 |
| 合作伙伴+战略 | 4 | $0 | P7 |
| 门禁+发布 (P7) | 3 | $0 | P7 |
| AGI 集成协议 | 8 | $500 (API) | P8 |
| 协议规范+可持续性 | 12 | $500 (专家) | P8 |
| 门禁+发布 (P8) | 10 | $0 | P8 |
| **合计** | **80** | **$2,500** | |

## 9. 验收标准

- [ ] MedicalCausalSDK: 医疗因果链准确率 ≥85%
- [ ] LegalComplianceSDK: 法律因果分析准确率 ≥80%
- [ ] EngineeringSafetySDK: 安全推理准确率 ≥80%
- [ ] MCIDomainSDK: 统一接口 3 领域
- [ ] AuditableCausalReasoning: 100% 推理有审计轨迹
- [ ] ComplianceRuleEngine: 3 领域规则集可用
- [ ] EdgeCloudHybrid: 边缘 <10ms, 云端兜底 <200ms
- [ ] AutoScaler: 100→1000 QPS <5s
- [ ] MCIPluginInterface: 规范 + ≥2 示例插件
- [ ] AGIIntegrationProtocol: 5 端点可用 + REST/gRPC

## 依赖关系

- **前置**: Ch07 (经济成本: TCO 分析), Ch09 (替代性评估: 行业定位), Ch15 (SocialCognition: SDK 社会协商), Ch16 (NeuralSymbolicFusion: AGI 协议需要融合推理)
- **被依赖**: 无 (终局章节，P8 完成后项目收束)
