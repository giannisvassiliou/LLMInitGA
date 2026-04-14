"""
FIXED: LLM-Enhanced Genetic Algorithms
Key improvements:
1. Constraint-aware initialization (prevents capacity violations)
2. Smarter adaptive mutations (fitness-trajectory based)
3. Enhanced LLM prompts with capacity guidance
4. Feedback loop for adaptive strategy
"""

import anthropic
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import time
from datetime import datetime
import os

# ============================================================================
# PROBLEM DEFINITIONS
# ============================================================================

@dataclass
class Item:
    id: int
    name: str
    value: int
    weight: int
    category: str
    subcategory: str
    priority_level: int  # 1=essential, 2=important, 3=nice-to-have

@dataclass
class SimpleItem:
    id: int
    value: int
    weight: int
    category: str

@dataclass
class SemanticItem:
    id: int
    name: str
    value: int
    weight: int
    category: str
    description: str
    requires: List[int]
    synergies: List[int]
    conflicts: List[int]
    enables: List[int]
    synergy_bonus: int = 0
    conflict_penalty: int = 0

# ============================================================================
# PROBLEM CLASSES
# ============================================================================

class SimpleProblem:
    """Easy problem for baseline comparison"""
    def __init__(self):
        categories = ['electronics', 'food', 'tools', 'clothing']
        self.items = []
        for i in range(50):
            cat = random.choice(categories)
            if cat == 'electronics':
                value = random.randint(50, 100)
                weight = random.randint(5, 15)
            elif cat == 'food':
                value = random.randint(20, 40)
                weight = random.randint(2, 8)
            elif cat == 'tools':
                value = random.randint(40, 80)
                weight = random.randint(10, 20)
            else:
                value = random.randint(15, 35)
                weight = random.randint(3, 10)
            
            self.items.append(SimpleItem(i, value, weight, cat))
        
        self.capacity = 150
        self.min_categories = 3
        self.problem_type = 'simple'
    
    def evaluate(self, solution):
        total_value = 0
        total_weight = 0
        categories = set()
        
        for i, selected in enumerate(solution):
            if selected:
                total_value += self.items[i].value
                total_weight += self.items[i].weight
                categories.add(self.items[i].category)
        
        penalties = 0
        if total_weight > self.capacity:
            penalties += (total_weight - self.capacity) * 10
        if len(categories) < self.min_categories:
            penalties += (self.min_categories - len(categories)) * 50
        
        fitness = total_value - penalties
        
        return fitness, {
            'value': total_value,
            'weight': total_weight,
            'categories': len(categories),
            'feasible': penalties == 0
        }

class HardKnapsackProblem:
    """Large-scale multi-category problem"""
    def __init__(self, n_items=200):
        self.items = self._generate_hard_items(n_items)
        self.capacity = 800
        self.min_categories = 6
        self.max_per_category = 25
        self.min_priority_1_items = 5
        self.target_weight_range = (700, 800)
        self.problem_type = 'hard'
        
    def _generate_hard_items(self, n_items):
        categories = {
            'electronics': {'value_range': (80, 200), 'weight_range': (10, 50)},
            'medical': {'value_range': (100, 250), 'weight_range': (2, 15)},
            'tools': {'value_range': (60, 150), 'weight_range': (15, 60)},
            'food': {'value_range': (20, 80), 'weight_range': (5, 40)},
            'clothing': {'value_range': (40, 120), 'weight_range': (5, 30)},
            'shelter': {'value_range': (80, 180), 'weight_range': (20, 70)},
            'navigation': {'value_range': (50, 140), 'weight_range': (1, 20)},
            'entertainment': {'value_range': (30, 100), 'weight_range': (3, 25)}
        }
        
        items = []
        items_per_category = n_items // len(categories)
        item_id = 0
        
        for cat_name, cat_info in categories.items():
            for i in range(items_per_category):
                base_value = random.randint(*cat_info['value_range'])
                base_weight = random.randint(*cat_info['weight_range'])
                
                # Trap items
                if random.random() < 0.15:
                    base_value = int(base_value * 1.3)
                    base_weight = int(base_weight * 1.8)
                
                # Priority levels
                if random.random() < 0.08:
                    priority = 1
                    base_value = int(base_value * 1.2)
                elif random.random() < 0.25:
                    priority = 2
                else:
                    priority = 3
                
                items.append(Item(
                    id=item_id,
                    name=f"{cat_name}_{i}",
                    value=base_value,
                    weight=base_weight,
                    category=cat_name,
                    subcategory=cat_name,
                    priority_level=priority
                ))
                item_id += 1
        
        # Fill remaining
        while len(items) < n_items:
            cat_name = random.choice(list(categories.keys()))
            cat_info = categories[cat_name]
            
            items.append(Item(
                id=item_id,
                name=f"{cat_name}_extra",
                value=random.randint(*cat_info['value_range']),
                weight=random.randint(*cat_info['weight_range']),
                category=cat_name,
                subcategory=cat_name,
                priority_level=random.choice([2, 3])
            ))
            item_id += 1
        
        return items
    
    def evaluate(self, solution: List[bool]) -> tuple:
        total_value = 0
        total_weight = 0
        categories = set()
        category_counts = {}
        priority_1_count = 0
        
        for i, selected in enumerate(solution):
            if selected:
                item = self.items[i]
                total_value += item.value
                total_weight += item.weight
                categories.add(item.category)
                category_counts[item.category] = category_counts.get(item.category, 0) + 1
                if item.priority_level == 1:
                    priority_1_count += 1
        
        # Hard constraints (define feasibility)
        hard_penalties = 0
        if total_weight > self.capacity:
            hard_penalties += (total_weight - self.capacity) * 20
        
        if len(categories) < self.min_categories:
            hard_penalties += (self.min_categories - len(categories)) * 150
        
        for cat, count in category_counts.items():
            if count > self.max_per_category:
                hard_penalties += (count - self.max_per_category) * 100
        
        if priority_1_count < self.min_priority_1_items:
            hard_penalties += (self.min_priority_1_items - priority_1_count) * 120
        
        # Soft constraint / preference: encourage packing into the target range,
        # but do NOT make the solution infeasible just for being underweight.
        soft_penalties = 0
        if total_weight < self.target_weight_range[0]:
            soft_penalties += (self.target_weight_range[0] - total_weight) * 2
        
        penalties = hard_penalties + soft_penalties
        fitness = total_value - penalties
        feasible = (hard_penalties == 0)
        
        return fitness, {
            'value': total_value,
            'weight': total_weight,
            'categories': len(categories),
            'category_counts': category_counts,
            'priority_1_count': priority_1_count,
            'hard_penalties': hard_penalties,
            'soft_penalties': soft_penalties,
            'penalties': penalties,
            'feasible': feasible
        }

# ============================================================================
# FIXED: LLM GUIDANCE WITH BETTER PROMPTS
# ============================================================================

def _safe_div(a, b, default=0.0):
    return a / b if b else default

def build_semantic_aggregate_summary(problem):
    """Build aggregate semantic summaries without exposing item-level identities to the LLM."""
    category_summary = {}
    total_weight = 0
    total_items = len(problem.items)

    dep_counts = {}
    dep_weight = {}
    conflict_counts = {}
    conflict_weight = {}
    synergy_counts = {}
    synergy_weight = {}

    requires_total = 0
    conflicts_total = 0
    synergies_total = 0
    synergy_bonus_total = 0
    conflict_penalty_total = 0
    participating_dep_items = 0
    participating_conflict_items = 0
    participating_synergy_items = 0

    for item in problem.items:
        cat = getattr(item, 'category', 'default')
        cs = category_summary.setdefault(cat, {
            'count': 0,
            'avg_value': 0.0,
            'avg_weight': 0.0,
            'requires_out': 0,
            'conflicts_out': 0,
            'synergies_out': 0,
            'avg_synergy_bonus': 0.0,
            'avg_conflict_penalty': 0.0
        })
        cs['count'] += 1
        cs['avg_value'] += item.value
        cs['avg_weight'] += item.weight
        cs['requires_out'] += len(getattr(item, 'requires', []) or [])
        cs['conflicts_out'] += len(getattr(item, 'conflicts', []) or [])
        cs['synergies_out'] += len(getattr(item, 'synergies', []) or [])
        cs['avg_synergy_bonus'] += int(getattr(item, 'synergy_bonus', 0) or 0)
        cs['avg_conflict_penalty'] += int(getattr(item, 'conflict_penalty', 0) or 0)
        total_weight += item.weight

        if getattr(item, 'requires', None):
            participating_dep_items += 1
        if getattr(item, 'conflicts', None):
            participating_conflict_items += 1
        if getattr(item, 'synergies', None):
            participating_synergy_items += 1

        for rid in getattr(item, 'requires', []) or []:
            if 0 <= rid < total_items:
                other_cat = getattr(problem.items[rid], 'category', 'default')
                dep_counts.setdefault(cat, {}).setdefault(other_cat, 0)
                dep_counts[cat][other_cat] += 1
                dep_weight.setdefault(cat, {}).setdefault(other_cat, 0)
                dep_weight[cat][other_cat] += item.weight
                requires_total += 1

        for cid in getattr(item, 'conflicts', []) or []:
            if 0 <= cid < total_items:
                other_cat = getattr(problem.items[cid], 'category', 'default')
                conflict_counts.setdefault(cat, {}).setdefault(other_cat, 0)
                conflict_counts[cat][other_cat] += 1
                conflict_weight.setdefault(cat, {}).setdefault(other_cat, 0)
                conflict_weight[cat][other_cat] += int(getattr(item, 'conflict_penalty', 0) or 0)
                conflicts_total += 1
                conflict_penalty_total += int(getattr(item, 'conflict_penalty', 0) or 0)

        for sid in getattr(item, 'synergies', []) or []:
            if 0 <= sid < total_items:
                other_cat = getattr(problem.items[sid], 'category', 'default')
                synergy_counts.setdefault(cat, {}).setdefault(other_cat, 0)
                synergy_counts[cat][other_cat] += 1
                synergy_weight.setdefault(cat, {}).setdefault(other_cat, 0)
                synergy_weight[cat][other_cat] += int(getattr(item, 'synergy_bonus', 0) or 0)
                synergies_total += 1
                synergy_bonus_total += int(getattr(item, 'synergy_bonus', 0) or 0)

    for cat, cs in category_summary.items():
        c = cs['count']
        cs['avg_value'] = round(_safe_div(cs['avg_value'], c), 2)
        cs['avg_weight'] = round(_safe_div(cs['avg_weight'], c), 2)
        cs['avg_synergy_bonus'] = round(_safe_div(cs['avg_synergy_bonus'], c), 2)
        cs['avg_conflict_penalty'] = round(_safe_div(cs['avg_conflict_penalty'], c), 2)

    def normalize_matrix(counts, denominators, weight_lookup=None, avg_name='avg_strength'):
        out = {}
        cats = sorted(category_summary.keys())
        for src in cats:
            src_total = max(1, denominators.get(src, 0))
            row = {}
            for dst, cnt in sorted(counts.get(src, {}).items()):
                avg_strength = _safe_div((weight_lookup or {}).get(src, {}).get(dst, 0), cnt) if cnt else 0.0
                row[dst] = {
                    'share': round(cnt / src_total, 3),
                    'count': int(cnt),
                    avg_name: round(avg_strength, 2),
                }
            if row:
                out[src] = row
        return out

    denominators = {cat: category_summary[cat]['count'] for cat in category_summary}
    dependency_matrix = normalize_matrix(dep_counts, denominators, dep_weight, avg_name='avg_trigger_weight')
    conflict_matrix = normalize_matrix(conflict_counts, denominators, conflict_weight, avg_name='avg_penalty')
    synergy_matrix = normalize_matrix(synergy_counts, denominators, synergy_weight, avg_name='avg_bonus')

    avg_item_weight = _safe_div(total_weight, total_items, 1.0)
    semantic_stats = {
        'items': total_items,
        'avg_item_weight': round(avg_item_weight, 2),
        'dependency_participation_rate': round(_safe_div(participating_dep_items, total_items), 3),
        'conflict_participation_rate': round(_safe_div(participating_conflict_items, total_items), 3),
        'synergy_participation_rate': round(_safe_div(participating_synergy_items, total_items), 3),
        'avg_requires_per_item': round(_safe_div(requires_total, total_items), 3),
        'avg_conflicts_per_item': round(_safe_div(conflicts_total, total_items), 3),
        'avg_synergies_per_item': round(_safe_div(synergies_total, total_items), 3),
        'avg_synergy_bonus': round(_safe_div(synergy_bonus_total, max(1, synergies_total)), 2),
        'avg_conflict_penalty': round(_safe_div(conflict_penalty_total, max(1, conflicts_total)), 2),
    }

    return {
        'category_summary': category_summary,
        'semantic_stats': semantic_stats,
        'dependency_matrix': dependency_matrix,
        'conflict_matrix': conflict_matrix,
        'synergy_matrix': synergy_matrix,
        'avg_item_weight': avg_item_weight,
    }

class LLMGuidance:
    def __init__(self, api_key: str):
        self.model_name = "claude-sonnet-4-20250514"
        self.client = anthropic.Anthropic(api_key=api_key)
        self.call_count = 0
        self.total_cost = 0.0
        self.strategy_history = []  # Track what strategies were tried
    
    def get_initialization_guidance(self, problem) -> dict:
        """Get initialization guidance based on problem type"""

        if problem.problem_type == 'simple':
            return self._get_simple_guidance(problem)
        elif problem.problem_type == 'hard':
            return self._get_hard_guidance(problem)
        elif problem.problem_type == 'semantic':
            return self._get_semantic_guidance(problem)
        else:
            return self._get_default_guidance(problem)
    
    def _get_simple_guidance(self, problem: SimpleProblem) -> dict:
        """FIXED: Enhanced guidance for simple problem with capacity awareness"""
        
        category_summary = {}
        total_weight = 0
        total_items = 0
        
        for item in problem.items:
            if item.category not in category_summary:
                category_summary[item.category] = {'count': 0, 'avg_value': 0, 'avg_weight': 0}
            category_summary[item.category]['count'] += 1
            category_summary[item.category]['avg_value'] += item.value
            category_summary[item.category]['avg_weight'] += item.weight
            total_weight += item.weight
            total_items += 1
        
        for cat in category_summary:
            count = category_summary[cat]['count']
            category_summary[cat]['avg_value'] /= count
            category_summary[cat]['avg_weight'] /= count
        
        avg_item_weight = total_weight / total_items
        # FIXED: Calculate safe selection probability
        safe_prob = (problem.capacity * 0.85) / (len(problem.items) * avg_item_weight)
        
        prompt = f"""You are guiding a genetic algorithm for a knapsack problem.

PROBLEM:
- {len(problem.items)} items across {len(category_summary)} categories
- Capacity: {problem.capacity} units (HARD CONSTRAINT - violations cost 10 points per unit!)
- Must include items from at least {problem.min_categories} categories

CATEGORY SUMMARY:
{json.dumps(category_summary, indent=2)}

CRITICAL CAPACITY GUIDANCE:
- Average item weight: {avg_item_weight:.1f} units
- Safe selection probability: ≤ {safe_prob:.3f} to avoid capacity violations
- Expected weight at {safe_prob:.3f}: {len(problem.items) * safe_prob * avg_item_weight:.0f} units

TASK: Provide initialization strategy that RESPECTS CAPACITY.

Respond in JSON:
{{
  "selection_probability": 0.XX (MUST be ≤ {safe_prob:.3f}),
  "category_priorities": {{"electronics": X.X, "food": X.X, "tools": X.X, "clothing": X.X}},
  "min_value_weight_ratio": X.X,
  "reasoning": "brief explanation focusing on capacity management"
}}"""

        self.call_count += 1
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.total_cost += 0.002
        
        response_text = message.content[0].text
        
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            guidance = json.loads(response_text[json_start:json_end])
            
            # FIXED: Safety clamp on selection probability
            if guidance['selection_probability'] > safe_prob:
                guidance['selection_probability'] = safe_prob * 0.9
            
            return guidance
        except:
            return {
                "selection_probability": min(0.12, safe_prob * 0.9),
                "category_priorities": {cat: 1.0 for cat in category_summary.keys()},
                "min_value_weight_ratio": 2.0,
                "reasoning": "fallback to safe defaults"
            }
    
    def _get_hard_guidance(self, problem: HardKnapsackProblem) -> dict:
        """FIXED: Enhanced guidance for hard problem with capacity awareness"""
        
        category_summary = {}
        total_weight = 0
        total_items = 0
        
        for item in problem.items:
            if item.category not in category_summary:
                category_summary[item.category] = {
                    'count': 0, 'avg_value': 0, 'avg_weight': 0, 'priority_1_count': 0
                }
            category_summary[item.category]['count'] += 1
            category_summary[item.category]['avg_value'] += item.value
            category_summary[item.category]['avg_weight'] += item.weight
            if item.priority_level == 1:
                category_summary[item.category]['priority_1_count'] += 1
            total_weight += item.weight
            total_items += 1
        
        for cat in category_summary:
            count = category_summary[cat]['count']
            category_summary[cat]['avg_value'] /= count
            category_summary[cat]['avg_weight'] /= count
        
        avg_item_weight = total_weight / total_items
        # FIXED: Calculate safe selection probability
        safe_prob = (problem.capacity * 0.85) / (len(problem.items) * avg_item_weight)
        
        prompt = f"""You are guiding a GA for a HARD knapsack with many constraints.

PROBLEM:
- {len(problem.items)} items across {len(category_summary)} categories
- Capacity: {problem.capacity} units (violations cost 20 points per unit - VERY EXPENSIVE!)
- Must select from {problem.min_categories}+ categories
- Max {problem.max_per_category} items per category
- Must include {problem.min_priority_1_items}+ priority-1 (essential) items
- Target weight: {problem.target_weight_range[0]}-{problem.target_weight_range[1]}

CATEGORY SUMMARY:
{json.dumps(category_summary, indent=2)}

CRITICAL CAPACITY GUIDANCE:
- Average item weight: {avg_item_weight:.1f} units
- Maximum safe base probability: {safe_prob:.3f}
- With priority multipliers (1.5x), effective probability will be HIGHER
- Expected weight formula: {len(problem.items)} × base_prob × multipliers × {avg_item_weight:.1f}
- To hit target {problem.target_weight_range[1]} units, keep base_prob LOW

CONSTRAINT PENALTIES (avoid these!):
- Over capacity: 20 points per unit over
- Missing categories: 150 points per missing category
- Too many per category: 100 points per excess item
- Missing priority-1: 120 points per missing item

TASK: Design initialization strategy that RESPECTS ALL CONSTRAINTS.

Respond in JSON:
{{
  "selection_probability": 0.XX (recommend ≤ {safe_prob * 0.7:.3f} to allow for multipliers),
  "category_priorities": {{"electronics": X.X, "medical": X.X, ...}} (values 0.8-1.5),
  "priority_multipliers": {{"1": X.X, "2": X.X, "3": X.X}} (keep modest: 1.2-1.5 max),
  "min_value_weight_ratio": X.X,
  "reasoning": "brief explanation of capacity management strategy"
}}"""

        self.call_count += 1
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.total_cost += 0.003
        
        response_text = message.content[0].text
        
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            guidance = json.loads(response_text[json_start:json_end])
            
            # FIXED: Safety clamp on selection probability
            if guidance['selection_probability'] > safe_prob * 0.7:
                guidance['selection_probability'] = safe_prob * 0.7
            
            # FIXED: Clamp multipliers to prevent overloading
            if 'priority_multipliers' in guidance:
                for key in guidance['priority_multipliers']:
                    if guidance['priority_multipliers'][key] > 1.5:
                        guidance['priority_multipliers'][key] = 1.5
            
            return guidance
        except:
            return {
                "selection_probability": min(0.08, safe_prob * 0.7),
                "category_priorities": {cat: 1.0 for cat in category_summary.keys()},
                "priority_multipliers": {"1": 1.3, "2": 1.1, "3": 1.0},
                "min_value_weight_ratio": 3.0,
                "reasoning": "fallback to conservative safe defaults"
            }
    
    
    def _get_semantic_guidance(self, problem) -> dict:
        """LLM guidance for semantic knapsack using aggregate relational summaries."""

        agg = build_semantic_aggregate_summary(problem)
        category_summary = agg['category_summary']
        semantic_stats = agg['semantic_stats']
        dependency_matrix = agg['dependency_matrix']
        conflict_matrix = agg['conflict_matrix']
        synergy_matrix = agg['synergy_matrix']
        avg_item_weight = agg['avg_item_weight']

        safe_prob = (problem.capacity * 0.85) / (len(problem.items) * max(avg_item_weight, 1e-9))
        capped_prob = min(0.12, safe_prob * 0.9)

        prompt = f"""You are guiding a genetic algorithm for a SEMANTIC knapsack problem.

PROBLEM:
- {len(problem.items)} items across {len(category_summary)} categories
- Capacity: {problem.capacity} units (violations cost 10 points per unit)
- Must include items from at least {problem.min_categories} categories
- Feasibility depends on capacity, category coverage, requirements, and conflicts

SEMANTIC GLOBAL STATS:
{json.dumps(semantic_stats, indent=2)}

CATEGORY SUMMARY:
{json.dumps(category_summary, indent=2)}

DEPENDENCY MATRIX (aggregate category-level requirements):
{json.dumps(dependency_matrix, indent=2)}

CONFLICT MATRIX (aggregate category-level incompatibilities):
{json.dumps(conflict_matrix, indent=2)}

SYNERGY MATRIX (aggregate category-level complementarities):
{json.dumps(synergy_matrix, indent=2)}

CRITICAL CAPACITY GUIDANCE:
- Average item weight: {avg_item_weight:.2f} units
- Maximum safe base probability: {safe_prob:.3f}
- Higher structure/synergy bias can increase effective selection pressure

TASK:
Design aggregate initialization biases for a greedy initializer.
Do NOT select items. Do NOT output item IDs.
Use the matrices to encourage dependency-compatible and synergy-compatible category combinations,
while avoiding conflict-heavy category combinations and respecting capacity.

Respond in JSON only:
{{
  "selection_probability": 0.XX,
  "category_priorities": {{"category_name": X.X, "...": X.X}},
  "min_value_weight_ratio": X.X,
  "dependency_bias": {{
    "enable": true,
    "cross_category_boost": X.X,
    "completion_importance": X.X,
    "max_missing_requirements_tolerance": X.X
  }},
  "conflict_avoidance": {{
    "enable": true,
    "cross_category_penalty": X.X,
    "intra_category_penalty": X.X,
    "strictness": X.X
  }},
  "synergy_bias": {{
    "enable": true,
    "cross_category_bonus": X.X,
    "intra_category_bonus": X.X,
    "bonus_cap": X.X
  }},
  "structure_vs_capacity_balance": X.X,
  "reasoning": "brief explanation"
}}
"""

        self.call_count += 1
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}]
        )
        self.total_cost += 0.0035
        response_text = message.content[0].text

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            guidance = json.loads(response_text[json_start:json_end])
        except Exception:
            guidance = {}

        guidance.setdefault('selection_probability', capped_prob)
        guidance['selection_probability'] = min(float(guidance.get('selection_probability', capped_prob)), capped_prob)
        guidance.setdefault('category_priorities', {cat: 1.0 for cat in category_summary.keys()})
        guidance.setdefault('min_value_weight_ratio', 2.5)
        guidance.setdefault('dependency_bias', {
            'enable': True,
            'cross_category_boost': 1.20,
            'completion_importance': 1.15,
            'max_missing_requirements_tolerance': 0.25,
        })
        guidance.setdefault('conflict_avoidance', {
            'enable': True,
            'cross_category_penalty': 1.25,
            'intra_category_penalty': 1.10,
            'strictness': 0.90,
        })
        guidance.setdefault('synergy_bias', {
            'enable': True,
            'cross_category_bonus': 1.15,
            'intra_category_bonus': 1.10,
            'bonus_cap': 1.50,
        })
        guidance.setdefault('structure_vs_capacity_balance', 0.60)
        guidance.setdefault('reasoning', 'fallback to aggregate semantic-aware defaults')

        # Clamp semantic parameters conservatively for stability
        dep = guidance['dependency_bias']
        dep['cross_category_boost'] = min(max(float(dep.get('cross_category_boost', 1.2)), 0.8), 1.8)
        dep['completion_importance'] = min(max(float(dep.get('completion_importance', 1.15)), 0.8), 1.8)
        dep['max_missing_requirements_tolerance'] = min(max(float(dep.get('max_missing_requirements_tolerance', 0.25)), 0.0), 1.0)

        conf = guidance['conflict_avoidance']
        conf['cross_category_penalty'] = min(max(float(conf.get('cross_category_penalty', 1.25)), 0.5), 2.5)
        conf['intra_category_penalty'] = min(max(float(conf.get('intra_category_penalty', 1.10)), 0.5), 2.5)
        conf['strictness'] = min(max(float(conf.get('strictness', 0.90)), 0.0), 1.0)

        syn = guidance['synergy_bias']
        syn['cross_category_bonus'] = min(max(float(syn.get('cross_category_bonus', 1.15)), 0.5), 2.0)
        syn['intra_category_bonus'] = min(max(float(syn.get('intra_category_bonus', 1.10)), 0.5), 2.0)
        syn['bonus_cap'] = min(max(float(syn.get('bonus_cap', 1.50)), 1.0), 3.0)

        guidance['structure_vs_capacity_balance'] = min(max(float(guidance.get('structure_vs_capacity_balance', 0.60)), 0.0), 1.0)
        return guidance

    def _get_default_guidance(self, problem) -> dict:
        """Default guidance"""
        return {
            "selection_probability": 0.10,
            "category_priorities": {},
            "min_value_weight_ratio": 2.0,
            "reasoning": "default configuration"
        }
    
    def get_adaptive_strategy(self, problem_type: str, generation: int, 
                            stats: dict, fitness_history: List[float]) -> str:
        """FIXED: Smarter adaptive strategy with feedback loop"""
        
        # Calculate improvement trends
        recent_improvement = 0
        long_improvement = 0
        
        if len(fitness_history) >= 20:
            recent_improvement = fitness_history[-1] - fitness_history[-20]
        if len(fitness_history) >= 50:
            long_improvement = fitness_history[-1] - fitness_history[-50]
        
        # Determine if we're making progress
        is_improving = recent_improvement > 10
        is_stagnant = stats['stagnant_gens'] > 15
        gens_remaining = 150 - generation
        
        # Simple heuristic decision (no LLM needed for basic cases)
        if is_improving:
            return 'continue'  # Making progress, don't interfere
        
        if not is_stagnant:
            return 'continue'  # Not stagnant yet
        
        # Only use LLM for complex decisions
        if generation % 40 == 0 and is_stagnant:  # Less frequent LLM calls
            
            # Build context with previous strategy results
            strategy_context = ""
            if self.strategy_history:
                last_strategy = self.strategy_history[-1]
                strategy_context = f"\nLAST STRATEGY: {last_strategy['action']} at gen {last_strategy['generation']}\n"
                strategy_context += f"Result: fitness {last_strategy['before_fitness']:.0f} → {fitness_history[-1]:.0f} ({fitness_history[-1] - last_strategy['before_fitness']:+.0f})\n"
            
            prompt = f"""GA checkpoint (gen {generation}/150, {problem_type}):

CURRENT STATE:
- Best fitness: {stats['best_fitness']:.0f}
- Avg fitness: {stats['avg_fitness']:.0f}
- Diversity (std): {stats['diversity']:.0f}
- Stagnant: {stats['stagnant_gens']} generations
- Feasible solutions: {stats['feasible_pct']:.0f}%

PROGRESS TRENDS:
- Last 20 gens: {recent_improvement:+.0f} fitness change
- Last 50 gens: {long_improvement:+.0f} fitness change
- Generations remaining: {gens_remaining}
{strategy_context}
CONTEXT:
- If recent_improvement > 10: Making good progress → Continue
- If stagnant < 15 gens: Not stuck yet → Continue
- If stagnant > 30 and diversity < 100: Stuck in local optimum → Need diversity
- If feasible < 60%: Too many invalid solutions → Decrease mutation
- If gens_remaining < 30: Focus on exploitation → Decrease mutation

Based on the TRENDS and CONTEXT above, recommend:
A) Increase mutation (if need more exploration, diversity high)
B) Decrease mutation (if need fine-tuning, nearing end)
C) Add immigrants (if stuck in local optimum with low diversity)
D) Continue (if making progress or strategy unclear)

Reply: ONE letter + reason (max 25 words explaining your logic)"""

            self.call_count += 1
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            self.total_cost += 0.001
            
            response = message.content[0].text.strip()
            
            # Record this strategy attempt
            strategy_decision = 'continue'
            if 'A' in response[:3]:
                strategy_decision = 'increase_mutation'
            elif 'B' in response[:3]:
                strategy_decision = 'decrease_mutation'
            elif 'C' in response[:3]:
                strategy_decision = 'add_immigrants'
            
            self.strategy_history.append({
                'generation': generation,
                'action': strategy_decision,
                'before_fitness': stats['best_fitness'],
                'reasoning': response
            })
            
            return strategy_decision
        else:
            # Fallback heuristic (no LLM)
            if stats['feasible_pct'] < 60:
                return 'decrease_mutation'  # Too many invalid solutions
            elif gens_remaining < 30:
                return 'decrease_mutation'  # Focus on exploitation
            elif stats['diversity'] < 100 and is_stagnant:
                return 'add_immigrants'  # Need diversity
            else:
                return 'continue'


# ============================================================================
# GUIDANCE CACHING (K-GUIDANCE ROTATION)
# ============================================================================

PROMPT_VERSION = "v1.2-semantic-aggregates-2026-03-23"

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _guidance_cache_key(problem_type: str,
                        dataset: str,
                        model_name: str,
                        hard_path: str,
                        semantic_path: str) -> str:
    dataset = (dataset or "synthetic").lower()
    file_hash = "none"
    file_id = dataset
    if dataset == "hard_json":
        file_id = os.path.basename(hard_path)
        if os.path.exists(hard_path):
            file_hash = _sha256_file(hard_path)
    elif dataset == "semantic_json":
        file_id = os.path.basename(semantic_path)
        if os.path.exists(semantic_path):
            file_hash = _sha256_file(semantic_path)
    else:
        # Synthetic: no external file, keep it stable by problem type only
        file_id = f"synthetic:{problem_type}"
        file_hash = "synthetic"

    return f"{PROMPT_VERSION}|{model_name}|{problem_type}|{dataset}|{file_id}|sha256={file_hash}"

def get_guidance_pool(llm: "LLMGuidance",
                      problem,
                      dataset: str,
                      k: int,
                      cache_file: str,
                      hard_path: str,
                      semantic_path: str) -> list:
    """Get K initialization guidances (generate once, then rotate across runs).

    Cache format: { cache_key: [guidance_dict, ...] }
    """
    k = int(k or 0)
    if k <= 0:
        return []

    cache = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    key = _guidance_cache_key(problem.problem_type, dataset, llm.model_name, hard_path, semantic_path)
    pool = cache.get(key, [])

    # Ensure pool is a list of dicts
    if not isinstance(pool, list):
        pool = []

    # If not enough cached, generate the missing ones
    missing = max(0, k - len(pool))
    for _ in range(missing):
        pool.append(llm.get_initialization_guidance(problem))

    # Persist updated cache
    if missing > 0 and cache_file:
        cache[key] = pool
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    return pool[:k]

# ============================================================================
# FIXED: GENETIC ALGORITHM WITH CONSTRAINT-AWARE INITIALIZATION
# ============================================================================

class GeneticAlgorithm:
    def __init__(self, problem, config: dict):
        self.problem = problem
        self.config = config
        self.population_size = config.get('population_size', 100)
        self.generations = config.get('generations', 150)
        self.mutation_rate = config.get('mutation_rate', 0.05)
        self.elite_size = config.get('elite_size', 10)
    
    def initialize_population_random(self):
        """Random initialization"""
        population = []
        p = 0.15
        
        for _ in range(self.population_size):
            solution = [random.random() < p for _ in range(len(self.problem.items))]
            population.append(solution)
        
        return population
    


    # ------------------------------------------------------------------------
    # NEW: Strong non-LLM heuristic initializer baseline (FFGR)
    # Feasibility-First Greedy + Randomized Repair/Diversification
    # ------------------------------------------------------------------------
    def initialize_population_ffgr(self):
        """FFGR initialization (non-LLM baseline).

        Constructs an initial population using a feasibility-first greedy heuristic
        (value/weight with modest priority boost), then applies constraint-aware
        repairs and light randomization for diversity.

        For semantic problems, it is conservative about requirements/conflicts.
        """
        n = len(self.problem.items)
        population = []

        min_categories = int(getattr(self.problem, 'min_categories', 0) or 0)
        max_per_category = int(getattr(self.problem, 'max_per_category', 10**9) or 10**9)
        min_priority_1 = int(getattr(self.problem, 'min_priority_1_items', 0) or 0)
        capacity = int(getattr(self.problem, 'capacity', 0) or 0)

        # modest handcrafted multipliers (kept small so this remains a fair baseline)
        p_mult = {1: 1.25, 2: 1.10, 3: 1.00}

        is_semantic = (getattr(self.problem, 'problem_type', '') == 'semantic')

        def ratio(item):
            r = item.value / max(getattr(item, 'weight', 1), 1)
            if hasattr(item, 'priority_level'):
                r *= p_mult.get(int(getattr(item, 'priority_level', 3) or 3), 1.0)
            return r

        def can_add_basic(i, current_weight, category_counts):
            item = self.problem.items[i]
            cat = getattr(item, 'category', 'default')
            if current_weight + item.weight > capacity:
                return False
            if category_counts.get(cat, 0) >= max_per_category:
                return False
            return True

        def semantic_can_add(i, used, current_weight, category_counts):
            # conservative: ensure capacity + conflicts, and satisfy requirements if possible
            item = self.problem.items[i]
            if not can_add_basic(i, current_weight, category_counts):
                return False, []
            # conflicts
            conf = set(getattr(item, 'conflicts', []) or [])
            if conf.intersection(used):
                return False, []
            for j in used:
                if i in set(getattr(self.problem.items[j], 'conflicts', []) or []):
                    return False, []
            # requirements
            reqs = list(getattr(item, 'requires', []) or [])
            to_add = []
            tmp_weight = current_weight + item.weight
            tmp_counts = dict(category_counts)

            for r in reqs:
                if r in used:
                    continue
                if not (0 <= r < n):
                    return False, []
                if not can_add_basic(r, tmp_weight, tmp_counts):
                    return False, []
                r_item = self.problem.items[r]
                # requirement conflicts with current used?
                if set(getattr(r_item, 'conflicts', []) or []).intersection(used):
                    return False, []
                if i in set(getattr(r_item, 'conflicts', []) or []):
                    return False, []
                # apply
                to_add.append(r)
                tmp_weight += r_item.weight
                r_cat = getattr(r_item, 'category', 'default')
                tmp_counts[r_cat] = tmp_counts.get(r_cat, 0) + 1
            return True, to_add

        for pop_idx in range(self.population_size):
            sol = [False] * n
            used = set()
            current_weight = 0
            categories = set()
            category_counts = {}
            priority_1_count = 0

            # score + tiny noise for diversity
            scored = []
            for i, item in enumerate(self.problem.items):
                s = ratio(item) * (0.95 + random.random() * 0.10)
                scored.append((s, random.random(), i))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

            # greedy build
            skip_p = 0.03 + (pop_idx / max(self.population_size, 1)) * 0.05
            for s, tie, i in scored:
                if random.random() < skip_p:
                    continue
                item = self.problem.items[i]
                cat = getattr(item, 'category', 'default')

                if is_semantic:
                    ok, reqs = semantic_can_add(i, used, current_weight, category_counts)
                    if not ok:
                        continue
                    # add reqs first
                    for r in reqs:
                        r_item = self.problem.items[r]
                        r_cat = getattr(r_item, 'category', 'default')
                        if not can_add_basic(r, current_weight, category_counts):
                            break
                        sol[r] = True
                        used.add(r)
                        current_weight += r_item.weight
                        categories.add(r_cat)
                        category_counts[r_cat] = category_counts.get(r_cat, 0) + 1
                    else:
                        sol[i] = True
                        used.add(i)
                        current_weight += item.weight
                        categories.add(cat)
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                else:
                    if not can_add_basic(i, current_weight, category_counts):
                        continue
                    sol[i] = True
                    used.add(i)
                    current_weight += item.weight
                    categories.add(cat)
                    category_counts[cat] = category_counts.get(cat, 0) + 1

                if hasattr(item, 'priority_level') and int(getattr(item, 'priority_level', 3) or 3) == 1:
                    priority_1_count += 1

            # repair for hard problems (min categories / min priority-1)
            if (not is_semantic) and (getattr(self.problem, 'problem_type', '') == 'hard'):
                if len(categories) < min_categories or priority_1_count < min_priority_1:
                    for s, tie, i in scored:
                        if sol[i]:
                            continue
                        item = self.problem.items[i]
                        cat = getattr(item, 'category', 'default')
                        need_cat = (len(categories) < min_categories and cat not in categories)
                        need_p1 = (priority_1_count < min_priority_1 and hasattr(item, 'priority_level') and int(item.priority_level) == 1)
                        if not (need_cat or need_p1):
                            continue
                        if not can_add_basic(i, current_weight, category_counts):
                            continue
                        sol[i] = True
                        used.add(i)
                        current_weight += item.weight
                        categories.add(cat)
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                        if hasattr(item, 'priority_level') and int(item.priority_level) == 1:
                            priority_1_count += 1
                        if len(categories) >= min_categories and priority_1_count >= min_priority_1:
                            break

            # light diversification (avoid for semantic to keep deps stable)
            if not is_semantic:
                n_moves = int(n * (0.08 + (pop_idx / max(self.population_size, 1)) * 0.07))
                for _ in range(n_moves):
                    j = random.randrange(n)
                    item = self.problem.items[j]
                    cat = getattr(item, 'category', 'default')
                    if sol[j]:
                        cat_count = category_counts.get(cat, 0)
                        would_break_cat = (cat_count == 1 and min_categories > 0 and len(categories) <= min_categories)
                        would_break_p1 = (hasattr(item, 'priority_level') and int(getattr(item, 'priority_level', 3) or 3) == 1
                                          and min_priority_1 > 0 and priority_1_count <= min_priority_1)
                        if would_break_cat or would_break_p1:
                            continue
                        sol[j] = False
                        used.discard(j)
                        current_weight -= item.weight
                        if cat_count <= 1:
                            category_counts.pop(cat, None)
                            categories.discard(cat)
                        else:
                            category_counts[cat] = cat_count - 1
                        if hasattr(item, 'priority_level') and int(getattr(item, 'priority_level', 3) or 3) == 1:
                            priority_1_count -= 1
                    else:
                        if not can_add_basic(j, current_weight, category_counts):
                            continue
                        sol[j] = True
                        used.add(j)
                        current_weight += item.weight
                        categories.add(cat)
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                        if hasattr(item, 'priority_level') and int(getattr(item, 'priority_level', 3) or 3) == 1:
                            priority_1_count += 1

            # semantic final cleanup if needed
            if is_semantic:
                fitness, st = self.problem.evaluate(sol)
                guard = 0
                while not st.get('feasible', False) and guard < n:
                    selected = [i for i, b in enumerate(sol) if b]
                    if not selected:
                        break
                    worst = min(selected, key=lambda i: ratio(self.problem.items[i]))
                    sol[worst] = False
                    fitness, st = self.problem.evaluate(sol)
                    guard += 1

            population.append(sol)

        return population
    def initialize_population_llm_guided(self, guidance: dict):
        """Constraint-aware LLM-guided initialization with semantic aggregate support."""
        population = []
        is_semantic = (getattr(self.problem, 'problem_type', '') == 'semantic')
        semantic_agg = build_semantic_aggregate_summary(self.problem) if is_semantic else None

        def can_add_basic(idx, current_weight, category_counts, max_per_category):
            item = self.problem.items[idx]
            cat = getattr(item, 'category', 'default')
            if current_weight + item.weight > self.problem.capacity:
                return False
            if category_counts.get(cat, 0) >= max_per_category:
                return False
            return True

        def semantic_requirement_plan(idx, used, current_weight, category_counts, max_per_category):
            item = self.problem.items[idx]
            reqs = list(getattr(item, 'requires', []) or [])
            missing = [r for r in reqs if r not in used and 0 <= r < len(self.problem.items)]
            req_weight = sum(self.problem.items[r].weight for r in missing)
            tmp_weight = current_weight
            tmp_counts = dict(category_counts)
            planned = []
            for r in missing:
                r_item = self.problem.items[r]
                r_cat = getattr(r_item, 'category', 'default')
                if tmp_weight + r_item.weight > self.problem.capacity:
                    return False, []
                if tmp_counts.get(r_cat, 0) >= max_per_category:
                    return False, []
                # avoid requirement conflicts with current used or with item
                if set(getattr(r_item, 'conflicts', []) or []).intersection(used):
                    return False, []
                if idx in set(getattr(r_item, 'conflicts', []) or []):
                    return False, []
                planned.append(r)
                tmp_weight += r_item.weight
                tmp_counts[r_cat] = tmp_counts.get(r_cat, 0) + 1
            return True, planned

        def semantic_live_multiplier(item, categories, selected_ids):
            mult = 1.0
            dep = guidance.get('dependency_bias', {})
            conf = guidance.get('conflict_avoidance', {})
            syn = guidance.get('synergy_bias', {})
            dep_matrix = semantic_agg['dependency_matrix'] if semantic_agg else {}
            conf_matrix = semantic_agg['conflict_matrix'] if semantic_agg else {}
            syn_matrix = semantic_agg['synergy_matrix'] if semantic_agg else {}
            cat = getattr(item, 'category', 'default')

            if dep.get('enable', False):
                for dst, meta in dep_matrix.get(cat, {}).items():
                    if dst in categories:
                        mult *= 1.0 + meta.get('share', 0.0) * (float(dep.get('cross_category_boost', 1.2)) - 1.0)
                reqs = list(getattr(item, 'requires', []) or [])
                if reqs:
                    req_hits = sum(1 for r in reqs if r in selected_ids)
                    mult *= 1.0 + _safe_div(req_hits, len(reqs)) * (float(dep.get('completion_importance', 1.15)) - 1.0)

            if syn.get('enable', False):
                syn_mult = 1.0
                for dst, meta in syn_matrix.get(cat, {}).items():
                    if dst in categories:
                        bonus_factor = float(syn.get('cross_category_bonus', 1.15)) if dst != cat else float(syn.get('intra_category_bonus', 1.10))
                        syn_mult *= 1.0 + meta.get('share', 0.0) * (bonus_factor - 1.0)
                mult *= min(syn_mult, float(syn.get('bonus_cap', 1.50)))

            if conf.get('enable', False):
                strictness = float(conf.get('strictness', 0.9))
                for dst, meta in conf_matrix.get(cat, {}).items():
                    if dst in categories:
                        penalty = float(conf.get('cross_category_penalty', 1.25)) if dst != cat else float(conf.get('intra_category_penalty', 1.10))
                        mult /= 1.0 + meta.get('share', 0.0) * (penalty - 1.0) * strictness
            return mult

        for pop_idx in range(self.population_size):
            solution = [False] * len(self.problem.items)
            current_weight = 0
            categories = set()
            category_counts = {}
            priority_1_count = 0
            selected_ids = set()

            if self.problem.problem_type == 'hard':
                min_categories = getattr(self.problem, 'min_categories', 0)
                max_per_category = getattr(self.problem, 'max_per_category', 999)
                min_priority_1 = getattr(self.problem, 'min_priority_1_items', 0)
            else:
                min_categories = getattr(self.problem, 'min_categories', 0)
                max_per_category = 999
                min_priority_1 = 0

            item_scores = []
            structure_balance = float(guidance.get('structure_vs_capacity_balance', 0.60))
            for i, item in enumerate(self.problem.items):
                base_prob = float(guidance.get('selection_probability', 0.1))
                cat_mult = float(guidance.get('category_priorities', {}).get(getattr(item, 'category', 'default'), 1.0))
                priority_mult = float(guidance.get('priority_multipliers', {}).get(str(getattr(item, 'priority_level', 3)), 1.0)) if hasattr(item, 'priority_level') else 1.0
                ratio = item.value / max(item.weight, 1)
                ratio_threshold = float(guidance.get('min_value_weight_ratio', 2.0))
                ratio_mult = 1.3 if ratio > ratio_threshold else 1.0
                lightweight_bonus = 1.0 + (1.0 - structure_balance) * (1.0 / max(item.weight, 1))
                semantic_prior = 1.0
                if is_semantic:
                    semantic_prior *= semantic_live_multiplier(item, set(), set())
                    if getattr(item, 'synergy_bonus', 0):
                        semantic_prior *= 1.0 + structure_balance * min(0.25, item.synergy_bonus / 200.0)
                    if getattr(item, 'conflict_penalty', 0):
                        semantic_prior /= 1.0 + structure_balance * min(0.35, item.conflict_penalty / 300.0)
                score = ratio * base_prob * cat_mult * priority_mult * ratio_mult * lightweight_bonus * semantic_prior
                score *= (0.9 + random.random() * 0.2)
                item_scores.append((i, score, item))
            item_scores.sort(key=lambda x: x[1], reverse=True)

            for item_idx, _, item in item_scores:
                cat = getattr(item, 'category', 'default')
                dynamic_mult = semantic_live_multiplier(item, categories, selected_ids) if is_semantic else 1.0
                if dynamic_mult < 0.55:
                    continue
                if not can_add_basic(item_idx, current_weight, category_counts, max_per_category):
                    continue

                planned = []
                if is_semantic:
                    # direct conflict checks
                    if set(getattr(item, 'conflicts', []) or []).intersection(selected_ids):
                        continue
                    if any(item_idx in set(getattr(self.problem.items[j], 'conflicts', []) or []) for j in selected_ids):
                        continue
                    ok, planned = semantic_requirement_plan(item_idx, selected_ids, current_weight + item.weight, {**category_counts, cat: category_counts.get(cat,0)+1}, max_per_category)
                    reqs = list(getattr(item, 'requires', []) or [])
                    missing_count = sum(1 for r in reqs if r not in selected_ids)
                    tolerance = float(guidance.get('dependency_bias', {}).get('max_missing_requirements_tolerance', 0.25))
                    if reqs and (missing_count / max(1, len(reqs))) > tolerance and not ok:
                        continue
                    if not ok and reqs:
                        continue

                solution[item_idx] = True
                current_weight += item.weight
                category_counts[cat] = category_counts.get(cat, 0) + 1
                categories.add(cat)
                selected_ids.add(item_idx)
                if hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1:
                    priority_1_count += 1

                for rid in planned:
                    if solution[rid]:
                        continue
                    r_item = self.problem.items[rid]
                    r_cat = getattr(r_item, 'category', 'default')
                    solution[rid] = True
                    current_weight += r_item.weight
                    category_counts[r_cat] = category_counts.get(r_cat, 0) + 1
                    categories.add(r_cat)
                    selected_ids.add(rid)

            if len(categories) < min_categories or priority_1_count < min_priority_1:
                for item_idx, _, item in item_scores:
                    if solution[item_idx]:
                        continue
                    cat = getattr(item, 'category', 'default')
                    need_category = (cat not in categories and len(categories) < min_categories)
                    need_priority = (hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1 and priority_1_count < min_priority_1)
                    if not (need_category or need_priority):
                        continue
                    if not can_add_basic(item_idx, current_weight, category_counts, max_per_category):
                        continue
                    solution[item_idx] = True
                    current_weight += item.weight
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    categories.add(cat)
                    selected_ids.add(item_idx)
                    if hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1:
                        priority_1_count += 1
                    if len(categories) >= min_categories and priority_1_count >= min_priority_1:
                        break

            diversity_factor = 0.06 if is_semantic else (0.1 + (pop_idx / self.population_size) * 0.1)
            n_swaps = int(len(solution) * diversity_factor)
            for _ in range(n_swaps):
                idx = random.randint(0, len(solution) - 1)
                item = self.problem.items[idx]
                cat = getattr(item, 'category', 'default')
                if solution[idx]:
                    cat_count = category_counts.get(cat, 0)
                    would_lose_category = (cat_count == 1 and len(categories) <= min_categories)
                    would_lose_priority = (hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1 and priority_1_count <= min_priority_1)
                    if is_semantic:
                        dependents_selected = any(idx in (getattr(self.problem.items[j], 'requires', []) or []) for j in selected_ids if j != idx)
                        if dependents_selected:
                            continue
                    if not would_lose_category and not would_lose_priority:
                        solution[idx] = False
                        current_weight -= item.weight
                        selected_ids.discard(idx)
                        if cat_count <= 1:
                            category_counts.pop(cat, None)
                            categories.discard(cat)
                        else:
                            category_counts[cat] = cat_count - 1
                        if hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1:
                            priority_1_count -= 1
                else:
                    if not can_add_basic(idx, current_weight, category_counts, max_per_category):
                        continue
                    if is_semantic and (set(getattr(item, 'conflicts', []) or []).intersection(selected_ids) or any(idx in set(getattr(self.problem.items[j], 'conflicts', []) or []) for j in selected_ids)):
                        continue
                    solution[idx] = True
                    current_weight += item.weight
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    categories.add(cat)
                    selected_ids.add(idx)
                    if hasattr(item, 'priority_level') and getattr(item, 'priority_level', 3) == 1:
                        priority_1_count += 1

            if is_semantic:
                fitness, st = self.problem.evaluate(solution)
                guard = 0
                while not st.get('feasible', False) and guard < len(solution):
                    selected = [i for i, b in enumerate(solution) if b]
                    if not selected:
                        break
                    worst = min(selected, key=lambda i: (self.problem.items[i].value / max(self.problem.items[i].weight, 1)))
                    solution[worst] = False
                    fitness, st = self.problem.evaluate(solution)
                    guard += 1

            population.append(solution)
        return population

    def evaluate_population(self, population):
        results = []
        for solution in population:
            fitness, stats = self.problem.evaluate(solution)
            results.append((fitness, stats))
        return results
    
    def tournament_selection(self, population, fitnesses, k=3):
        tournament_idx = random.sample(range(len(population)), k)
        tournament = [(population[i], fitnesses[i]) for i in tournament_idx]
        winner = max(tournament, key=lambda x: x[1])
        return winner[0].copy()
    
    def two_point_crossover(self, parent1, parent2):
        point1 = random.randint(0, len(parent1)-1)
        point2 = random.randint(point1, len(parent1)-1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def mutate(self, solution, rate=None):
        if rate is None:
            rate = self.mutation_rate
        
        return [not gene if random.random() < rate else gene for gene in solution]
    
    def run(self, use_llm_init=False, llm_guidance=None, init_guidance=None, use_adaptive=False, verbose=False, use_ffgr_init=False):
        """Run GA with optional LLM guidance"""
        
        # Initialize
        if use_ffgr_init:
            population = self.initialize_population_ffgr()
            llm_object = None
        elif use_llm_init:
            # Allow supplying a precomputed guidance dict (e.g., from a cached pool),
            # while still using an LLM object for adaptive decisions if desired.
            if isinstance(init_guidance, dict):
                population = self.initialize_population_llm_guided(init_guidance)
                llm_object = llm_guidance if (use_adaptive and llm_guidance and not isinstance(llm_guidance, dict)) else None
            elif llm_guidance:
                if isinstance(llm_guidance, dict):
                    population = self.initialize_population_llm_guided(llm_guidance)
                    llm_object = None
                else:
                    llm_object = llm_guidance
                    guidance = llm_object.get_initialization_guidance(self.problem)
                    population = self.initialize_population_llm_guided(guidance)
            else:
                # No guidance provided → fallback to random init
                population = self.initialize_population_random()
                llm_object = None
        else:
            population = self.initialize_population_random()
            llm_object = None
        
        best_solution = None
        best_fitness = float('-inf')
        # Track best STRICTLY FEASIBLE solution separately
        best_feasible_solution = None
        best_feasible_fitness = float('-inf')
        feasible_best_history = []
        fitness_history = []
        stagnant_count = 0
        
        current_mutation_rate = self.mutation_rate
        
        for gen in range(self.generations):
            # Evaluate
            eval_results = self.evaluate_population(population)
            fitnesses = [f for f, _ in eval_results]
            stats_list = [s for _, s in eval_results]
            
            # Track best (overall)
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
                stagnant_count = 0
            else:
                stagnant_count += 1

            # Track best STRICTLY FEASIBLE
            feasible_indices = [i for i, s in enumerate(stats_list) if s.get('feasible', False)]
            if feasible_indices:
                gen_best_feas_idx = max(feasible_indices, key=lambda i: fitnesses[i])
                gen_best_feas_fitness = fitnesses[gen_best_feas_idx]
                if gen_best_feas_fitness > best_feasible_fitness:
                    best_feasible_fitness = gen_best_feas_fitness
                    best_feasible_solution = population[gen_best_feas_idx].copy()

            fitness_history.append(best_fitness)
            feasible_best_history.append(best_feasible_fitness if best_feasible_solution is not None else float('nan'))
            
            # FIXED: Smarter adaptive strategy
            if use_adaptive and llm_object and gen > 0 and gen % 20 == 0:
                feasible_count = sum(1 for s in stats_list if s.get('feasible', False))
                pop_stats = {
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitnesses),
                    'diversity': np.std(fitnesses),
                    'stagnant_gens': stagnant_count,
                    'feasible_pct': (feasible_count / len(stats_list)) * 100
                }
                
                strategy = llm_object.get_adaptive_strategy(
                    problem_type=self.problem.problem_type,
                    generation=gen,
                    stats=pop_stats,
                    fitness_history=fitness_history  # FIXED: Pass history
                )
                
                if strategy == 'increase_mutation':
                    current_mutation_rate = min(0.15, self.mutation_rate * 1.3)
                elif strategy == 'decrease_mutation':
                    current_mutation_rate = max(0.02, self.mutation_rate * 0.8)
                elif strategy == 'add_immigrants':
                    # FIXED: Only add immigrants if diversity is truly low
                    if pop_stats['diversity'] < 100:
                        n_immigrants = self.population_size // 20  # Reduced from 10%
                        # Replace worst performers with new random solutions
                        worst_indices = np.argsort(fitnesses)[:n_immigrants]
                        new_immigrants = self.initialize_population_random()[:n_immigrants]
                        for i, idx in enumerate(worst_indices):
                            population[idx] = new_immigrants[i]
                        continue
            
            # Selection and reproduction
            next_population = []
            
            # Elitism
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            next_population.extend([population[i].copy() for i in elite_indices])
            
            # Generate offspring
            while len(next_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                child1, child2 = self.two_point_crossover(parent1, parent2)
                
                child1 = self.mutate(child1, current_mutation_rate)
                child2 = self.mutate(child2, current_mutation_rate)
                
                next_population.extend([child1, child2])
            
            population = next_population[:self.population_size]
            
            if verbose and gen % 30 == 0:
                print(f"  Gen {gen}: Best={best_fitness:.0f}, Avg={np.mean(fitnesses):.0f}, Stagnant={stagnant_count}")
        
        # Final evaluation (prefer strictly feasible best)
        if best_feasible_solution is not None:
            final_fitness, final_stats = self.problem.evaluate(best_feasible_solution)
            best_solution_out = best_feasible_solution
            best_fitness_out = best_feasible_fitness
        else:
            # No feasible solution found in this run
            final_fitness, final_stats = self.problem.evaluate(best_solution)
            final_stats = dict(final_stats)
            final_stats['feasible'] = False
            best_solution_out = best_solution
            best_fitness_out = float('nan')
        
        return {
            'best_solution': best_solution_out,
            'best_fitness': best_fitness_out,
            'best_overall_fitness': best_fitness,
            'best_feasible_fitness': (best_feasible_fitness if best_feasible_solution is not None else float('nan')),
            'fitness_history': fitness_history,
            'feasible_best_history': feasible_best_history,
            'final_stats': final_stats
        }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment_on_problem(problem, problem_name: str, api_key: str, n_runs: int = 5,
                             dataset: str = "synthetic",
                             cache_k: int = 5,
                             cache_file: str = "llm_guidance_cache.json",
                             hard_path: str = "hard_knapsack_200.json",
                             semantic_path: str = "semantic_expedition.json"):
    """Run complete experiment on one problem type"""

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {problem_name}")
    print(f"{'='*70}")

    llm = LLMGuidance(api_key)
    effective_cache_k = max(1, cache_k)
    if cache_k <= 0:
        print("[fix] cache_k <= 0 would make LLM-init fall back to random init; forcing cache_k=1.")
    guidance_pool = get_guidance_pool(
        llm, problem, dataset=dataset, k=effective_cache_k,
        cache_file=cache_file, hard_path=hard_path,
        semantic_path=semantic_path
    )
    if not guidance_pool:
        raise RuntimeError("LLM guidance pool is empty; aborting to avoid mislabeled 'LLM Init' runs that actually use random initialization.")

    results = {
        'pure_ga': [],
        'ffgr_init': [],
        'llm_init': [],
        'llm_init_adaptive': []
    }

    times = {k: [] for k in results}
    feasibility_rates = {k: [] for k in results}

    config = {
        'population_size': 100,
        'generations': 150,
        'mutation_rate': 0.05,
        'elite_size': 10
    }

    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")

        # 1. Pure GA
        print("  Running Pure GA...")
        ga = GeneticAlgorithm(problem, config)
        start_time = time.time()
        result_pure = ga.run(use_llm_init=False, verbose=False)
        time_pure = time.time() - start_time

        results['pure_ga'].append(result_pure['best_fitness'])
        times['pure_ga'].append(time_pure)
        feasibility_rates['pure_ga'].append(result_pure['final_stats']['feasible'])
        print(f"    Fitness: {result_pure['best_fitness']:.0f}, Time: {time_pure:.1f}s, Feasible: {result_pure['final_stats']['feasible']}")

        # 2. FFGR heuristic initialization (strong non-LLM baseline)
        print("  Running FFGR-Init GA...")
        ga = GeneticAlgorithm(problem, config)
        start_time = time.time()
        result_ffgr = ga.run(use_llm_init=False, use_ffgr_init=True, verbose=False)
        time_ffgr = time.time() - start_time

        results['ffgr_init'].append(result_ffgr['best_fitness'])
        times['ffgr_init'].append(time_ffgr)
        feasibility_rates['ffgr_init'].append(result_ffgr['final_stats']['feasible'])
        print(f"    Fitness: {result_ffgr['best_fitness']:.0f}, Time: {time_ffgr:.1f}s, Feasible: {result_ffgr['final_stats']['feasible']}")
        print(f"    Improvement vs Pure: {result_ffgr['best_fitness'] - result_pure['best_fitness']:.0f}")

        # 3. LLM-guided initialization
        print("  Running LLM-Init GA...")
        ga = GeneticAlgorithm(problem, config)
        start_time = time.time()
        init_g = guidance_pool[run % len(guidance_pool)] if guidance_pool else None
        result_llm_init = ga.run(use_llm_init=True, llm_guidance=None, init_guidance=init_g, verbose=False)
        time_llm_init = time.time() - start_time

        results['llm_init'].append(result_llm_init['best_fitness'])
        times['llm_init'].append(time_llm_init)
        feasibility_rates['llm_init'].append(result_llm_init['final_stats']['feasible'])
        print(f"    Fitness: {result_llm_init['best_fitness']:.0f}, Time: {time_llm_init:.1f}s, Feasible: {result_llm_init['final_stats']['feasible']}")
        print(f"    Improvement vs Pure: {result_llm_init['best_fitness'] - result_pure['best_fitness']:.0f}")

        # 4. LLM init + adaptive
        print("  Running LLM-Init + Adaptive GA...")
        ga = GeneticAlgorithm(problem, config)
        llm_adaptive = LLMGuidance(api_key)  # Fresh instance to avoid shared history
        start_time = time.time()
        init_g = guidance_pool[run % len(guidance_pool)] if guidance_pool else None
        result_adaptive = ga.run(use_llm_init=True, llm_guidance=llm_adaptive, init_guidance=init_g,
                                 use_adaptive=True, verbose=False)
        time_adaptive = time.time() - start_time

        results['llm_init_adaptive'].append(result_adaptive['best_fitness'])
        times['llm_init_adaptive'].append(time_adaptive)
        feasibility_rates['llm_init_adaptive'].append(result_adaptive['final_stats']['feasible'])
        print(f"    Fitness: {result_adaptive['best_fitness']:.0f}, Time: {time_adaptive:.1f}s, Feasible: {result_adaptive['final_stats']['feasible']}")
        print(f"    Improvement vs Pure: {result_adaptive['best_fitness'] - result_pure['best_fitness']:.0f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {problem_name}")
    print(f"{'='*70}")

    def _finite(vals):
        return [v for v in vals if not (isinstance(v, float) and np.isnan(v))]

    for method in ['pure_ga', 'ffgr_init', 'llm_init', 'llm_init_adaptive']:
        fitness_values = results[method]  # NaN if no feasible solution in that run
        time_values = times[method]
        feasible = feasibility_rates[method]

        finite = _finite(fitness_values)

        print(f"\n{method.upper().replace('_', ' ')}:")
        if finite:
            print(f"  Fitness (best feasible): {np.mean(finite):.1f} ± {np.std(finite):.1f}")
            print(f"  Range (feasible): [{np.min(finite):.0f}, {np.max(finite):.0f}]")
        else:
            print("  Fitness (best feasible): n/a")
            print("  Range (feasible): n/a")
        print(f"  Time: {np.mean(time_values):.1f}s ± {np.std(time_values):.1f}s")
        print(f"  Feasibility: {sum(feasible)}/{len(feasible)} runs")

    baseline_mean = np.mean(_finite(results['pure_ga'])) if _finite(results['pure_ga']) else float('nan')
    ffgr_mean = np.mean(_finite(results['ffgr_init'])) if _finite(results['ffgr_init']) else float('nan')
    llm_init_mean = np.mean(_finite(results['llm_init'])) if _finite(results['llm_init']) else float('nan')
    adaptive_mean = np.mean(_finite(results['llm_init_adaptive'])) if _finite(results['llm_init_adaptive']) else float('nan')

    if np.isnan(baseline_mean) or baseline_mean == 0:
        improvement_ffgr = float('nan')
        improvement_init = float('nan')
        improvement_adaptive = float('nan')
    else:
        improvement_ffgr = ((ffgr_mean / baseline_mean) - 1) * 100
        improvement_init = ((llm_init_mean / baseline_mean) - 1) * 100
        improvement_adaptive = ((adaptive_mean / baseline_mean) - 1) * 100

    print(f"\nIMPROVEMENTS (vs Pure GA):")
    print(f"  FFGR Init: {improvement_ffgr:+.1f}%")
    print(f"  LLM Init: {improvement_init:+.1f}%")
    print(f"  LLM Init + Adaptive: {improvement_adaptive:+.1f}%")

    print(f"\nLLM USAGE (pool generation only):")
    print(f"  Total calls: {llm.call_count}")
    print(f"  Estimated cost: ${llm.total_cost:.3f}")
    print(f"  Cost per run: ${llm.total_cost / max(n_runs,1):.3f}")

    return {
        'results': results,
        'times': times,
        'feasibility': feasibility_rates,
        'improvements': {
            'ffgr_init': improvement_ffgr,
            'llm_init': improvement_init,
            'adaptive': improvement_adaptive
        },
        'llm_stats': {
            'calls': llm.call_count,
            'cost': llm.total_cost
        }
    }

# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test(api_key: str, dataset: str = "synthetic", run_easy: bool = False,
              cache_k: int = 0, cache_file: str = "llm_guidance_cache.json",
              hard_path: str = "hard_knapsack_200.json", semantic_path: str = "semantic_expedition.json"):
    """Quick test with 2 runs"""
    
    print("="*70)
    print("FIXED VERSION - Quick Test (2 runs per problem type)")
    print("Expected improvements:")
    print("  Easy: +2-5% (marginal due to small problem)")
    print("  Hard: +8-15% (significant due to constraint awareness)")
    print("="*70)
    
    if run_easy:
        print("\n1. Testing EASY problem (50 items)...")
        easy_problem = SimpleProblem()
        easy_results = run_experiment_on_problem(easy_problem, "Easy", api_key, n_runs=2,
                                         dataset="synthetic", cache_k=max(1, cache_k),
                                         cache_file=cache_file, hard_path=hard_path, semantic_path=semantic_path)
    else:
        easy_results = None

    # Run primary problem based on dataset choice
    dataset_key = (dataset or "synthetic").lower()
    if dataset_key == "semantic_json":
        print("\n2. Testing SEMANTIC problem (from semantic_expedition.json)...")
        problem_label = "Semantic"
    else:
        print("\n2. Testing HARD problem (200 items)...")
        problem_label = "Hard"

    hard_problem = make_problem_from_dataset(dataset, hard_path=hard_path, semantic_path=semantic_path)
    print(f"{problem_label.upper()} ITEMS:", len(hard_problem.items))

    hard_results = run_experiment_on_problem(hard_problem, problem_label, api_key, n_runs=2,
                                     dataset=dataset, cache_k=cache_k,
                                     cache_file=cache_file, hard_path=hard_path, semantic_path=semantic_path)
    
    print("\n\n" + "="*70)
    print("QUICK TEST RESULTS - FIXED VERSION")
    print("="*70)
    
    if easy_results is not None:
        print(f"\nEasy Problem:")
        print(f"  LLM improvement: {easy_results['improvements']['llm_init']:+.1f}%")
        print(f"  Conclusion: {'✓ LLM helps!' if easy_results['improvements']['llm_init'] > 2 else 'LLM adds marginal value (expected for easy problems)'}")
    else:
        print("\nEasy Problem: (skipped)")

    dataset_key = (dataset or "synthetic").lower()
    summary_label = "Semantic" if dataset_key == "semantic_json" else "Hard"
    print(f"\n{summary_label} Problem:")
    print(f"  LLM improvement: {hard_results['improvements']['llm_init']:+.1f}%")
    print(f"  Conclusion: {'✓✓ LLM helps significantly!' if hard_results['improvements']['llm_init'] > 8 else '✓ LLM helps moderately' if hard_results['improvements']['llm_init'] > 3 else '✗ LLM adds little value'}")
    
    total_cost = hard_results['llm_stats']['cost'] + (easy_results['llm_stats']['cost'] if easy_results is not None else 0.0)
    print(f"\nTotal cost: ${total_cost:.2f}")
    
    print("\n" + "="*70)
    print("KEY FIXES APPLIED:")
    print("="*70)
    print("1. ✓ Constraint-aware initialization (greedy + repair)")
    print("2. ✓ Enhanced LLM prompts with capacity guidance")
    print("3. ✓ Smarter adaptive mutations with feedback loop")
    print("4. ✓ Safety clamps on selection probabilities")
    print("5. ✓ Reduced immigrant injection (10% → 5%)")

# ============================================================================
# MAIN
# ============================================================================


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def _method_label(method):
    return {
        "pure_ga": "Pure GA",
        "llm_init": "LLM Init",
        "ffgr_init": "FFGR Init",
        "llm_init_adaptive": "LLM Init + Adaptive",
    }.get(method, method)

def plot_feasible_curves(dfc_problem, outdir, problem_name):
    # Plot mean best-feasible-by-generation curves
    gens = sorted(dfc_problem["generation"].unique())
    plt.figure()
    for method_key in ["pure_ga", "ffgr_init", "llm_init", "llm_init_adaptive"]:
        d = dfc_problem[dfc_problem["method"] == method_key]
        pivot = d.pivot_table(index="run", columns="generation",
                              values="best_feasible_fitness", aggfunc="first")
        mean = pivot.mean(axis=0, skipna=True)
        plt.plot(gens, [mean.get(g, float("nan")) for g in gens],
                 label=_method_label(method_key))

    plt.xlabel("Generation")
    plt.ylabel("Best feasible fitness")
    plt.title("Feasible best by generation — " + problem_name)
    plt.legend()
    plt.savefig(os.path.join(outdir, "feasible_curve_" + problem_name.lower() + ".png"),
                dpi=200, bbox_inches="tight")
    plt.close()

def camera_ready_experiments(api_key, outdir="camera_ready_results",
                             n_runs=30, seed0=12345, dataset: str = "synthetic",
                             run_easy: bool = False,
                             cache_k: int = 0, cache_file: str = "llm_guidance_cache.json",
                             hard_path: str = "hard_knapsack_200.json",
                             semantic_path: str = "semantic_expedition.json"):

    # Camera-ready experiments: fixed seeds, CSVs, feasible curves
    _ensure_outdir(outdir)

    total_pool_calls = 0
    total_pool_cost = 0.0

    hard_problem = make_problem_from_dataset(dataset, hard_path=hard_path, semantic_path=semantic_path)

    # By default, run only the primary (Hard/Semantic) problem.
    if str(dataset).lower() == "semantic_json":
        problems = [("Semantic", hard_problem)]
    else:
        problems = [("Hard", hard_problem)]
        if run_easy:
            easy_problem = SimpleProblem()
            problems.insert(0, ("Easy", easy_problem))
    methods = [
        ("pure_ga", dict(use_llm_init=False, use_adaptive=False)),
        ("ffgr_init", dict(use_llm_init=False, use_adaptive=False)),
        ("llm_init", dict(use_llm_init=True, use_adaptive=False)),
        ("llm_init_adaptive", dict(use_llm_init=True, use_adaptive=True)),
    ]

    all_rows = []
    curve_rows = []

    config = {
        "population_size": 100,
        "generations": 150,
        "mutation_rate": 0.05,
        "elite_size": 10
    }

    for problem_name, problem in problems:
        # Build (and cache) K different initialization guidances ONCE per problem,
        # then rotate them across runs for diversity without repeated LLM calls.
        llm_cache = LLMGuidance(api_key)
        pool_k = max(1, cache_k)
        if cache_k <= 0:
            print(f"[fix] cache_k <= 0 for {problem_name}; forcing cache_k=1 so LLM-init is real LLM-init.")
        guidance_pool = get_guidance_pool(
            llm_cache, problem, dataset=dataset, k=pool_k,
            cache_file=cache_file, hard_path=hard_path, semantic_path=semantic_path
        )
        if not guidance_pool:
            raise RuntimeError(f"LLM guidance pool is empty for {problem_name}; aborting to avoid mislabeled LLM-init results.")
        guidance_pool_calls = llm_cache.call_count
        guidance_pool_cost = llm_cache.total_cost
        total_pool_calls += guidance_pool_calls
        total_pool_cost += guidance_pool_cost

        for run_idx in range(n_runs):
            seed = seed0 + run_idx

            # Fresh instance per run so adaptive history doesn't leak between runs
            llm_adaptive = LLMGuidance(api_key)

            for method_key, flags in methods:
                _set_seed(seed)
                ga = GeneticAlgorithm(problem, config)

                t0 = time.time()
                if method_key == "pure_ga":
                    result = ga.run(use_llm_init=False, verbose=False)
                    calls, cost = 0, 0.0
                elif method_key == "ffgr_init":
                    result = ga.run(use_llm_init=False, use_ffgr_init=True, verbose=False)
                    calls, cost = 0, 0.0
                elif method_key == "llm_init":
                    init_g = guidance_pool[run_idx % len(guidance_pool)] if guidance_pool else None
                    result = ga.run(use_llm_init=True, llm_guidance=None,
                                    init_guidance=init_g, use_adaptive=False, verbose=False)
                    calls, cost = 0, 0.0
                else:
                    init_g = guidance_pool[run_idx % len(guidance_pool)] if guidance_pool else None
                    result = ga.run(use_llm_init=True, llm_guidance=llm_adaptive,
                                    init_guidance=init_g, use_adaptive=True, verbose=False)
                    calls, cost = llm_adaptive.call_count, llm_adaptive.total_cost

                elapsed = time.time() - t0
                final_stats = result.get("final_stats", {})
                feasible = bool(final_stats.get("feasible", False))
                best_feas = result.get("best_fitness", float("nan"))

                all_rows.append({
                    "problem": problem_name,
                    "method": method_key,
                    "run": run_idx,
                    "seed": seed,
                    "best_feasible_fitness": best_feas,
                    "feasible": feasible,
                    "time_sec": elapsed,
                    "llm_calls": calls,
                    "llm_cost_est": cost,
                })

                hist = result.get("feasible_best_history", [])
                for gen, val in enumerate(hist):
                    curve_rows.append({
                        "problem": problem_name,
                        "method": method_key,
                        "run": run_idx,
                        "seed": seed,
                        "generation": gen,
                        "best_feasible_fitness": val
                    })

        # Write after each problem
        pd.DataFrame(all_rows).to_csv(os.path.join(outdir, "summary_runs.csv"),
                                      index=False)
        pd.DataFrame(curve_rows).to_csv(
            os.path.join(outdir, "feasible_curves_long.csv"), index=False)

    # Plot curves
    dfc = pd.DataFrame(curve_rows)
    for pname in dfc["problem"].unique():
        plot_feasible_curves(dfc[dfc["problem"] == pname], outdir, pname)

    # Console summary
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 70)
    print("CAMERA-READY SUMMARY (best feasible fitness only)")
    print("=" * 70)
    print(f"\nOne-time init-guidance cache generation: calls={total_pool_calls}, cost_est=${total_pool_cost:.3f} (reused/rotated across runs)")

    for pname in df["problem"].unique():
        print("\n" + pname + ":")
        for method_key in ["pure_ga", "ffgr_init", "llm_init", "llm_init_adaptive"]:
            d = df[(df["problem"] == pname) & (df["method"] == method_key)]
            feas = d[d["feasible"] == True]
            mean = float(np.nanmean(feas["best_feasible_fitness"])) if len(feas) else float("nan")
            std = float(np.nanstd(feas["best_feasible_fitness"])) if len(feas) else float("nan")
            print(f"  {_method_label(method_key)}: "
                  f"mean={mean:.1f}, std={std:.1f}, "
                  f"feasible_runs={len(feas)}/{len(d)}, "
                  f"mean_time={d['time_sec'].mean():.2f}s")

# PATCH: REAL DATASET LOADERS (REPLACES SYNTHETIC GENERATION)
# ============================================================================

def load_hard_knapsack_from_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    items = []
    for it in data["items"]:
        items.append(Item(
            id=it["id"],
            name=it["name"],
            value=it["value"],
            weight=it["weight"],
            category=it["category"],
            subcategory=it.get("subcategory", it["category"]),
            priority_level=it.get("priority_level", 3)
        ))

    problem = HardKnapsackProblem.__new__(HardKnapsackProblem)
    problem.items = items
    problem.capacity = data["capacity"]
    problem.min_categories = data["min_categories"]
    problem.max_per_category = data["max_per_category"]
    problem.min_priority_1_items = 5
    problem.target_weight_range = (700, problem.capacity)
    problem.problem_type = "hard"

    return problem


class SemanticKnapsackProblem:
    def __init__(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        self.capacity = data["capacity"]
        self.min_categories = data["min_categories"]
        self.problem_type = "semantic"

        self.items = []
        for it in data["items"]:
            self.items.append(SemanticItem(
                id=it["id"],
                name=it["name"],
                value=it["value"],
                weight=it["weight"],
                category=it["category"],
                description=it.get("description", ""),
                requires=it.get("requires", []),
                synergies=it.get("synergies", []),
                conflicts=it.get("conflicts", []),
                enables=it.get("enables", []),
                synergy_bonus=it.get("synergy_bonus", 0),
                conflict_penalty=it.get("conflict_penalty", 0),
            ))

    def evaluate(self, solution):
        total_value = 0
        total_weight = 0
        penalties = 0
        used = {i for i, b in enumerate(solution) if b}
        categories = set()

        for i in used:
            item = self.items[i]
            total_value += item.value
            total_weight += item.weight
            categories.add(item.category)

            for r in item.requires:
                if r not in used:
                    penalties += 200

            for c in item.conflicts:
                if c in used:
                    penalties += item.conflict_penalty

            for s in item.synergies:
                if s in used:
                    total_value += item.synergy_bonus

        if total_weight > self.capacity:
            penalties += (total_weight - self.capacity) * 10

        if len(categories) < self.min_categories:
            penalties += (self.min_categories - len(categories)) * 100

        fitness = total_value - penalties
        feasible = penalties == 0

        return fitness, {
            "value": total_value,
            "weight": total_weight,
            "categories": len(categories),
            "penalties": penalties,
            "feasible": feasible
        }


# ============================================================================
# USAGE SWITCH (REPLACES SYNTHETIC CONSTRUCTION)
# ============================================================================
# Example:
# problem = load_hard_knapsack_from_json("hard_knapsack_200.json")
# problem = SemanticKnapsackProblem("semantic_expedition.json")



def make_problem_from_dataset(dataset: str,
                              hard_path: str = "hard_knapsack_200.json",
                              semantic_path: str = "semantic_expedition.json"):
    """Factory: returns a problem instance based on dataset choice.
    dataset:
      - 'synthetic' (default): original SimpleProblem + HardKnapsackProblem
      - 'hard_json': hard_knapsack_200.json (replaces HardKnapsackProblem)
      - 'semantic_json': semantic_expedition.json (semantic constraints)
    """
    dataset = (dataset or "synthetic").lower()
    if dataset == "hard_json":
        return load_hard_knapsack_from_json(hard_path)
    if dataset == "semantic_json":
        return SemanticKnapsackProblem(semantic_path)
    # fallback
    return HardKnapsackProblem()


def main():
    import argparse, os

    parser = argparse.ArgumentParser(
        description="LLM-Enhanced Genetic Algorithm (real datasets supported)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run quick test")
    group.add_argument("--camera", action="store_true", help="Run camera-ready experiments")

    parser.add_argument("api_key", nargs="?", default=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")

    parser.add_argument("--dataset",
                        choices=["synthetic", "hard_json", "semantic_json"],
                        default="hard_json",
                        help="Dataset source for HARD problem")

    parser.add_argument("--cache-k", type=int, default=5,
                        help="Number of distinct LLM initialization guidances to generate once and rotate across runs (0 disables)")
    parser.add_argument("--cache-file", type=str, default="llm_guidance_cache.json",
                        help="Path to JSON cache file for LLM initialization guidance pools")


    parser.add_argument(
        "--easy",
        action="store_true",
        help="Also run the Easy (synthetic) problem"
    )

    parser.add_argument("--runs", type=int, default=30, help="Number of runs (camera-ready)")
    parser.add_argument("--seed0", type=int, default=12345, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="camera_ready_results",
                        help="Output directory")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide as argument or set ANTHROPIC_API_KEY.")

    if args.quick:
        quick_test(args.api_key, dataset=args.dataset, run_easy=args.easy,
             cache_k=args.cache_k, cache_file=args.cache_file,
             hard_path="hard_knapsack_200.json", semantic_path="semantic_expedition.json")
    elif args.camera:
        camera_ready_experiments(
            args.api_key,
            outdir=args.outdir,
            n_runs=args.runs,
            seed0=args.seed0,
            dataset=args.dataset,
            run_easy=args.easy,
            cache_k=args.cache_k,
            cache_file=args.cache_file,
            hard_path="hard_knapsack_200.json",
            semantic_path="semantic_expedition.json"
        )


if __name__ == "__main__":
    main()