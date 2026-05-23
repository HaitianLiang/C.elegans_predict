from __future__ import annotations

import pandas as pd
from typing import Dict

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class CellFeatureExtractor:
    """细胞特征提取器（含三维组合特征 + 特征表格输出 + 分类开关控制）"""
    
    def __init__(self, feature_toggles: Dict[str, bool] = None):
        # 基础初始化
        self.dim_names = ['x', 'y', 'z']
        self.dim_indices = {name: i for i, name in enumerate(self.dim_names)}
        self.cell_relations = {
            'ABa': {'mother': 'AB', 'daughters': ['ABal', 'ABar'], 'mother_gi': 'G1', 'daughter_gi': 'G2', 'time_offset': 1},
            'ABp': {'mother': 'AB', 'daughters': ['ABpl', 'ABpr'], 'mother_gi': 'G1', 'daughter_gi': 'G2', 'time_offset': 1},
            'EMS': {'mother': 'P1', 'daughters': ['E', 'MS'], 'mother_gi': 'G2', 'daughter_gi': 'G3', 'time_offset': 1},
            'P2': {'mother': 'P1', 'daughters': ['C', 'P3'], 'mother_gi': 'G3', 'daughter_gi': 'G4', 'time_offset': 1},
            'ABal': {'mother': 'ABa', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'ABar': {'mother': 'ABa', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'ABpl': {'mother': 'ABp', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'ABpr': {'mother': 'ABp', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'E': {'mother': 'EMS', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'MS': {'mother': 'EMS', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'C': {'mother': 'P2', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0},
            'P3': {'mother': 'P2', 'daughters': [], 'mother_gi': '', 'daughter_gi': '', 'time_offset': 0}
        }
        self.mother_daughter_combinations = [f"{mother}-{daughter}" for mother, info in self.cell_relations.items() for daughter in info['daughters']]
        self.gi_to_t = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3}
        self.t_to_gi = {v: k for k, v in self.gi_to_t.items()}
        
        # 调试统计
        self.debug_stats = {
            'total_groups': 0, 'valid_combinations_found': 0,
            'missing_mother': 0, 'missing_daughter': 0,
            'wrong_gi_mother': 0, 'wrong_gi_daughter': 0,
            'empty_adj_matrix': 0, 'invalid_adj_strength': 0, 'processed_cells': 0
        }

        # 存储所有特征的元数据（用于生成表格）
        self.feature_metadata = []
        
        # ========== 特征分类开关配置 ==========
        # 默认所有特征分类都开启
        default_toggles = {
            'self_single_dim': True,          # 细胞自身的单维度
            'self_coupled_polynomial': True,  # 细胞自身的耦合多项式（二维+三维）
            'adjacent_polynomial': True,      # 与相邻细胞的多项式
            'self_other_functions': False,     # 细胞自身的其他函数（对数、三角、指数、倒数、激活函数）
            'adjacent_other_functions': False  # 与相邻细胞的其他函数（三角、指数、分数、根号、对数、激活函数）
        }
        # 合并用户自定义开关（优先级：用户输入 > 默认配置）
        self.feature_toggles = {**default_toggles, **(feature_toggles or {})}

    # ========== 1. 细胞自身的单维度特征 ==========
    def _self_single_dim_features(self, coords, prefix):
        """细胞自身的单维度特征（多项式1-3次项）"""
        if not self.feature_toggles['self_single_dim']:
            return [], []
        
        features = []
        names = []
        metadata = []
        for dim in self.dim_names:
            idx = self.dim_indices[dim]
            val = coords[idx]
            # 1次项
            feat_name = f"{prefix}{dim}"
            formula = f"${dim}$"
            features.append(val)
            names.append(feat_name)
            metadata.append(("细胞自身的单维度", feat_name, formula))
            # 2次项
            feat_name = f"{prefix}{dim}_{dim}"
            formula = f"${dim}^2$"
            features.append(val**2)
            names.append(feat_name)
            metadata.append(("细胞自身的单维度", feat_name, formula))
            # 3次项
            feat_name = f"{prefix}{dim}_{dim}_{dim}"
            formula = f"${dim}^3$"
            features.append(val**3)
            names.append(feat_name)
            metadata.append(("细胞自身的单维度", feat_name, formula))
        
        self.feature_metadata.extend(metadata)
        return features, names

    # ========== 2. 细胞自身的耦合多项式特征 ==========
    def _3d_dimension_coupling_features(self, coords, prefix):
        """细胞自身的三维耦合多项式特征（移除与二维重复的二次项×单变量）"""
        if not self.feature_toggles['self_coupled_polynomial']:
            return [], []

        x, y, z = coords[0], coords[1], coords[2]
        features = []
        names = []
        metadata = []

        # 1. 三维基础算术组合
        prod_xyz_formula = "$x \\times y \\times z$"
        features.append(x*y*z)
        names.append(f"{prefix}3d_couple_prod_xyz")
        metadata.append(("细胞自身的耦合多项式", names[-1], prod_xyz_formula))

        # 2. 乘积平方的扩展
        prod_sq_xyz_formula = "$(x \\times y \\times z)^2$"
        x2_yz_sq_formula = "$(x^2 \\times y \\times z)^2$"
        xy2_z_sq_formula = "$(x \\times y^2 \\times z)^2$"
        x_yz2_sq_formula = "$(x \\times y \\times z^2)^2$"

        features.extend([
            (x*y*z)**2,
            (x**2 * y * z)**2,
            (x * y**2 * z)** 2,
            (x * y * z**2)**2
        ])

        names.extend([
            f"{prefix}3d_couple_prod_sq_xyz",
            f"{prefix}3d_couple_prod_sq_x2yz",
            f"{prefix}3d_couple_prod_sq_xy2z",
            f"{prefix}3d_couple_prod_sq_xyz2"
        ])

        metadata.extend([
            ("细胞自身的耦合多项式", names[-4], prod_sq_xyz_formula),
            ("细胞自身的耦合多项式", names[-3], x2_yz_sq_formula),
            ("细胞自身的耦合多项式", names[-2], xy2_z_sq_formula),
            ("细胞自身的耦合多项式", names[-1], x_yz2_sq_formula)
        ])

        self.feature_metadata.extend(metadata)
        return features, names
    
    def _coupled_polynomial_functions(self, xi, xj, dim1, dim2, prefix):
        """二维耦合多项式子函数"""
        if not self.feature_toggles['self_coupled_polynomial']:
            return [], []
        
        features = []
        names = []
        metadata = []
        # 乘积
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xi_xj"
        formula = f"${dim1} \\times {dim2}$"
        features.append(xi * xj)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        # 差值平方
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xj_minus_xi_pow2"
        formula = f"$({dim2} - {dim1})^2$"
        features.append((xj - xi)**2)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        # 乘积平方
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xi_xj_pow2"
        formula = f"$({dim1} \\times {dim2})^2$"
        features.append((xi * xj)**2)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        # 3次项：xi²xj
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xi2_xj"
        formula = f"${dim1}^2 \\times {dim2}$"
        features.append(xi**2 * xj)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        # 3次项：xixj²
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xi_xj2"
        formula = f"${dim1} \\times {dim2}^2$"
        features.append(xi * xj**2)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        # 3次项：乘积立方
        feat_name = f"{prefix}2d_couple_{dim1}{dim2}_xi_xj_pow3"
        formula = f"$({dim1} \\times {dim2})^3$"
        features.append((xi * xj)**3)
        names.append(feat_name)
        metadata.append(("细胞自身的耦合多项式", feat_name, formula))
        
        self.feature_metadata.extend(metadata)
        return features, names

    def _self_coupled_polynomial_features(self, coords, prefix):
        """细胞自身的耦合多项式特征（二维+三维整合）"""
        if not self.feature_toggles['self_coupled_polynomial']:
            return [], []
        
        x, y, z = coords[0], coords[1], coords[2]
        features = []
        names = []
        dim_pairs = [('x', x, 'y', y), ('x', x, 'z', z), ('y', y, 'z', z)]

        # 二维耦合多项式
        for dim1, val1, dim2, val2 in dim_pairs:
            poly_feats, poly_names = self._coupled_polynomial_functions(val1, val2, dim1, dim2, prefix)
            features.extend(poly_feats)
            names.extend(poly_names)

        # 三维耦合多项式
        three_d_feats, three_d_names = self._3d_dimension_coupling_features(coords, prefix)
        features.extend(three_d_feats)
        names.extend(three_d_names)

        return features, names

    # ========== 3. 细胞自身的其他函数特征 ==========
    def _self_other_functions_features(self, coords, prefix):
        """细胞自身的其他函数特征（对数、三角、指数、倒数、激活函数）"""
        if not self.feature_toggles['self_other_functions']:
            return [], []
        
        features = []
        names = []
        metadata = []
        x, y, z = coords[0], coords[1], coords[2]
        dim_pairs = [('x', x, 'y', y), ('x', x, 'z', z), ('y', y, 'z', z)]

        # 3.1 单维度其他函数
        for dim in self.dim_names:
            idx = self.dim_indices[dim]
            val = coords[idx]
            
            # 对数特征（ln、log2、log10）
            safe_val = np.clip(np.abs(val), 1e-10, 1e10)
            features.extend([np.log(safe_val), np.log2(safe_val), np.log10(safe_val)])
            log_names = [f"{prefix}ln{dim}", f"{prefix}log2{dim}", f"{prefix}log10{dim}"]
            names.extend(log_names)
            metadata.extend([
                ("细胞自身的其他函数", log_names[0], f"$\\ln(|{dim}|)$"),
                ("细胞自身的其他函数", log_names[1], f"$\\log_2(|{dim}|)$"),
                ("细胞自身的其他函数", log_names[2], f"$\\log_{{10}}(|{dim}|)$")
            ])
            
            # 三角特征（sin、cos、tan）
            features.extend([np.sin(val), np.cos(val), np.clip(np.tan(val), -1e6, 1e6)])
            trig_names = [f"{prefix}sin_{dim}", f"{prefix}cos_{dim}", f"{prefix}tan_{dim}"]
            names.extend(trig_names)
            metadata.extend([
                ("细胞自身的其他函数", trig_names[0], f"$\\sin({dim})$"),
                ("细胞自身的其他函数", trig_names[1], f"$\\cos({dim})$"),
                ("细胞自身的其他函数", trig_names[2], f"$\\tan({dim})$（限幅[-1e6,1e6]）")
            ])
            
            # 指数特征（exp）
            features.append(np.exp(val))
            exp_name = f"{prefix}exp_{dim}"
            names.append(exp_name)
            metadata.append(("细胞自身的其他函数", exp_name, f"$\\exp({dim})$"))
            
            # 倒数特征
            safe_val = val if abs(val) > 1e-10 else 1e-10
            features.append(1.0 / safe_val)
            frac_name = f"{prefix}frac_{dim}"
            names.append(frac_name)
            metadata.append(("细胞自身的其他函数", frac_name, f"$1/{dim}$（{dim}≠0，否则取1e10）"))
            
            # 激活函数特征（sigmoid、tanh）
            features.extend([1/(1+np.exp(-val)), np.tanh(val)])
            act_names = [f"{prefix}sigmoid_{dim}", f"{prefix}tanh_{dim}"]
            names.extend(act_names)
            metadata.extend([
                ("细胞自身的其他函数", act_names[0], f"$\\frac{{1}}{{1 + \\exp(-{dim})}}$"),
                ("细胞自身的其他函数", act_names[1], f"$\\tanh({dim})$")
            ])

        # 3.2 二维耦合其他函数（三角、根号、对数、指数）
        for dim1, val1, dim2, val2 in dim_pairs:
            product = val1 * val2
            diff = val1 - val2
            
            # 三角耦合
            features.extend([np.sin(product), np.cos(diff)])
            trig_couple_names = [
                f"{prefix}2d_couple_{dim1}{dim2}_trig_sin_product",
                f"{prefix}2d_couple_{dim1}{dim2}_trig_cos_diff"
            ]
            names.extend(trig_couple_names)
            metadata.extend([
                ("细胞自身的其他函数", trig_couple_names[0], f"$\\sin({dim1} \\times {dim2})$"),
                ("细胞自身的其他函数", trig_couple_names[1], f"$\\cos({dim1} - {dim2})$")
            ])
            
            # 根号/对数/指数耦合
            sqrt_product = np.sqrt(np.abs(product))
            safe_product = np.clip(np.abs(product), 1e-10, 1e10)
            log_product = np.log(safe_product)
            exp_product = np.exp(product / 10)
            
            other_couple_names = [
                f"{prefix}2d_couple_{dim1}{dim2}_root_sqrt_product",
                f"{prefix}2d_couple_{dim1}{dim2}_log_ln_product",
                f"{prefix}2d_couple_{dim1}{dim2}_exp_product"
            ]
            features.extend([sqrt_product, log_product, exp_product])
            names.extend(other_couple_names)
            metadata.extend([
                ("细胞自身的其他函数", other_couple_names[0], f"$\\sqrt{{|{dim1} \\times {dim2}|}}$"),
                ("细胞自身的其他函数", other_couple_names[1], f"$\\ln(|{dim1} \\times {dim2}|)$"),
                ("细胞自身的其他函数", other_couple_names[2], f"$\\exp\\left(\\frac{{{dim1} \\times {dim2}}}{{10}}\\right)$")
            ])
            
            # 三维组合其他函数（三角、指数、对数、根号）
            sum_xyz = val1 + val2 + (z if dim1 != 'z' else y)
            prod_xyz = val1 * val2 * (z if dim1 != 'z' else y)
            features.extend([np.sin(sum_xyz), np.cos(prod_xyz), np.exp(sum_xyz/10), np.log(np.clip(np.abs(prod_xyz), 1e-10, 1e10)), np.sqrt(np.abs(sum_xyz))])
            three_d_other_names = [
                f"{prefix}3d_couple_trig_sin_sum_xyz",
                f"{prefix}3d_couple_trig_cos_prod_xyz",
                f"{prefix}3d_couple_exp_sum_xyz",
                f"{prefix}3d_couple_log_ln_prod_xyz",
                f"{prefix}3d_couple_root_sqrt_sum_xyz"
            ]
            names.extend(three_d_other_names)
            metadata.extend([
                ("细胞自身的其他函数", three_d_other_names[0], "$\\sin(x + y + z)$"),
                ("细胞自身的其他函数", three_d_other_names[1], "$\\cos(x \\times y \\times z)$"),
                ("细胞自身的其他函数", three_d_other_names[2], "$\\exp\\left(\\frac{x + y + z}{10}\\right)$"),
                ("细胞自身的其他函数", three_d_other_names[3], "$\\ln(|x \\times y \\times z|)$"),
                ("细胞自身的其他函数", three_d_other_names[4], "$\\sqrt{|x + y + z|}$")
            ])

        self.feature_metadata.extend(metadata)
        return features, names

    # ========== 4. 与相邻细胞的多项式特征 ==========
    def _adjacent_polynomial_features(self, xi, xj):
        """与相邻细胞的多项式特征（基础多项式耦合 + 新增项）"""
        if not self.feature_toggles['adjacent_polynomial']:
            return [], []
        
        features = []
        names = []
        metadata = []
        
        # 基础多项式耦合 + 新增项
        features.extend([
            xi * xj,
            (xj - xi),          # 新增：xj - xi 一次项
            (xj - xi)**2,
            (xj - xi)**3,       # 新增：xj - xi 三次项
            (xi * xj)**2,
            xi**2 * xj,
            xi * xj**2,
            (xi * xj)**3,
            xj,                 # 新增：xj 一次项
            xj**2,              # 新增：xj 二次项
            xj**3               # 新增：xj 三次项
        ])
        poly_names = [
            "xi_xj",
            "xj_minus_xi",
            "xj_minus_xi_pow2",
            "xj_minus_xi_pow3",
            "xi_xj_pow2",
            "xi2_xj",
            "xi_xj2",
            "xi_xj_pow3",
            "xj",
            "xj_pow2",
            "xj_pow3"
        ]
        names.extend(poly_names)
        metadata.extend([
            ("与相邻细胞的多项式", poly_names[0], "$x_i \\times x_j$"),
            ("与相邻细胞的多项式", poly_names[1], "$x_j - x_i$"),
            ("与相邻细胞的多项式", poly_names[2], "$(x_j - x_i)^2$"),
            ("与相邻细胞的多项式", poly_names[3], "$(x_j - x_i)^3$"),
            ("与相邻细胞的多项式", poly_names[4], "$(x_i \\times x_j)^2$"),
            ("与相邻细胞的多项式", poly_names[5], "$x_i^2 \\times x_j$"),
            ("与相邻细胞的多项式", poly_names[6], "$x_i \\times x_j^2$"),
            ("与相邻细胞的多项式", poly_names[7], "$(x_i \\times x_j)^3$"),
            ("与相邻细胞的多项式", poly_names[8], "$x_j$"),
            ("与相邻细胞的多项式", poly_names[9], "$x_j^2$"),
            ("与相邻细胞的多项式", poly_names[10], "$x_j^3$")
        ])
        
        self.feature_metadata.extend(metadata)
        return features, names

    # ========== 5. 与相邻细胞的其他函数特征 ==========
    def _adjacent_other_functions_features(self, xi, xj):
        """与相邻细胞的其他函数特征（三角、指数、分数、根号、对数、激活函数）"""
        if not self.feature_toggles['adjacent_other_functions']:
            return [], []
        
        features = []
        names = []
        metadata = []
        
        # 三角耦合
        trig_feats = [np.sin(xi*xj), xi*np.sin(xj), np.cos(xi*xj), xi*np.cos(xj)]
        trig_names = ['sin_xi_xj', 'xi_sin_xj', 'cos_xi_xj', 'xi_cos_xj']
        features.extend(trig_feats)
        names.extend(trig_names)
        metadata.extend([
            ("与相邻细胞的其他函数", trig_names[0], "$\\sin(x_i \\times x_j)$"),
            ("与相邻细胞的其他函数", trig_names[1], "$x_i \\times \\sin(x_j)$"),
            ("与相邻细胞的其他函数", trig_names[2], "$\\cos(x_i \\times x_j)$"),
            ("与相邻细胞的其他函数", trig_names[3], "$x_i \\times \\cos(x_j)$")
        ])
        
        # 指数耦合
        exp_feats = [np.exp(xi*xj), np.exp(xj-xi), xi*np.exp(xj)]
        exp_names = ['exp_xi_xj', 'exp_xj_Minus_xi', 'xi_exp_xj']
        features.extend(exp_feats)
        names.extend(exp_names)
        metadata.extend([
            ("与相邻细胞的其他函数", exp_names[0], "$\\exp(x_i \\times x_j)$"),
            ("与相邻细胞的其他函数", exp_names[1], "$\\exp(x_j - x_i)$"),
            ("与相邻细胞的其他函数", exp_names[2], "$x_i \\times \\exp(x_j)$")
        ])
        
        # 分数耦合
        xj_safe = xj if abs(xj) > 1e-5 else 1e-5
        xixj_safe = (xi*xj) if abs(xi*xj) > 1e-5 else 1e-5
        frac_feats = [1/xixj_safe, xi/xj_safe]
        frac_names = ['frac_xi_xj', 'xi_frac_xj']
        features.extend(frac_feats)
        names.extend(frac_names)
        metadata.extend([
            ("与相邻细胞的其他函数", frac_names[0], "$1/(x_i \\times x_j)$（非零安全处理）"),
            ("与相邻细胞的其他函数", frac_names[1], "$x_i/x_j$（非零安全处理）")
        ])
        
        # 根号耦合
        root_feats = [np.sqrt(np.abs(xi))*np.sqrt(np.abs(xj)), np.cbrt(xi)*np.cbrt(xj)]
        root_names = ['root2_xi_root2_xj', 'root3_xi_root3_xj']
        features.extend(root_feats)
        names.extend(root_names)
        metadata.extend([
            ("与相邻细胞的其他函数", root_names[0], "$\\sqrt{|x_i|} \\times \\sqrt{|x_j|}$"),
            ("与相邻细胞的其他函数", root_names[1], "$\\sqrt[3]{x_i} \\times \\sqrt[3]{x_j}$")
        ])
        
        # 对数耦合
        safe_xi = np.clip(np.abs(xi), 1e-10, 1e10)
        safe_xj = np.clip(np.abs(xj), 1e-10, 1e10)
        log_feats = [np.log(safe_xi)*np.log(safe_xj)]
        log_names = ['ln_xi_ln_xj']
        features.extend(log_feats)
        names.extend(log_names)
        metadata.append(("与相邻细胞的其他函数", log_names[0], "$\\ln(|x_i|) \\times \\ln(|x_j|)$"))
        
        # 激活函数耦合
        # Sigmoid特征
        sigmoid_feats = [
            1/(1+np.exp(-xi*xj)),
            1/(1+np.exp(-(xj-xi))),
            xi/(1+np.exp(-xj))
        ]
        sigmoid_names = ['sigmoid_xi_xj', 'sigmoid_xj_Minus_xi', 'xi_sigmoid_xj']
        # Tanh特征
        tanh_feats = [np.tanh(xi*xj), np.tanh(xj-xi), xi*np.tanh(xj)]
        tanh_names = ['tanh_xi_xj', 'tanh_xj_Minus_xi', 'xi_tanh_xj']
        # Hill特征
        hill_feats = [
            (xi*xj**g)/(xi*xj**g + 1) for g in [1,2,3]
        ] + [
            ((xj-xi)**g)/((xj-xi)**g + 1) for g in [1,2,3]
        ]
        hill_names = [f'hill_xi_xj_{g}' for g in [1,2,3]] + [f'hill_xj_Minus_xi_{g}' for g in [1,2,3]]
        
        act_feats = sigmoid_feats + tanh_feats + hill_feats
        act_names = sigmoid_names + tanh_names + hill_names
        features.extend(act_feats)
        names.extend(act_names)
        
        # 激活函数元数据
        act_metadata = [
            ("与相邻细胞的其他函数", sigmoid_names[0], "$\\frac{1}{1 + \\exp(-x_i x_j)}$"),
            ("与相邻细胞的其他函数", sigmoid_names[1], "$\\frac{1}{1 + \\exp(-(x_j - x_i))}$"),
            ("与相邻细胞的其他函数", sigmoid_names[2], "$\\frac{x_i}{1 + \\exp(-x_j)}$"),
            ("与相邻细胞的其他函数", tanh_names[0], "$\\tanh(x_i x_j)$"),
            ("与相邻细胞的其他函数", tanh_names[1], "$\\tanh(x_j - x_i)$"),
            ("与相邻细胞的其他函数", tanh_names[2], "$x_i \\tanh(x_j)$")
        ]
        for g in [1,2,3]:
            act_metadata.append(("与相邻细胞的其他函数", f'hill_xi_xj_{g}', "$\\frac{(x_i x_j)^g}{(x_i x_j)^g + 1}$"))
            act_metadata.append(("与相邻细胞的其他函数", f'hill_xj_Minus_xi_{g}', "$\\frac{(x_j - x_i)^g}{(x_j - x_i)^g + 1}$"))
        metadata.extend(act_metadata)
        
        self.feature_metadata.extend(metadata)
        return features, names

    # ========== 邻接矩阵处理（整合相邻细胞特征） ==========
    def _sum_with_adj_cells(self, current_cell_name: str, current_cell_coords: np.ndarray, 
                           current_time_cells: Dict[str, Dict], adj_matrix: Dict[str, float], 
                           prefix: str) -> Tuple[List[float], List[str]]:
        features = []
        names = []
        metadata = []
        total_strength = 0.0
        coupled_feats_per_dim = {dim: [] for dim in self.dim_names}

        for neighbor_name, strength in adj_matrix.items():
            if strength <= 0 or neighbor_name not in current_time_cells:
                continue
            neighbor_coords = current_time_cells[neighbor_name]['self_coords']
            total_strength += strength

            for dim in self.dim_names:
                xi = current_cell_coords[self.dim_indices[dim]]
                xj = neighbor_coords[self.dim_indices[dim]]
                
                # 相邻细胞多项式特征
                poly_feats, _ = self._adjacent_polynomial_features(xi, xj)
                # 相邻细胞其他函数特征
                other_feats, _ = self._adjacent_other_functions_features(xi, xj)
                
                # 加权合并
                weighted_feats = [f * strength for f in poly_feats + other_feats]
                coupled_feats_per_dim[dim].append(weighted_feats)

        # 维度内耦合特征（整合多项式+其他函数）
        _, sample_poly_names = self._adjacent_polynomial_features(0, 0)
        _, sample_other_names = self._adjacent_other_functions_features(0, 0)
        sample_coupled_names = sample_poly_names + sample_other_names
        
        for dim in self.dim_names:
            if coupled_feats_per_dim[dim]:
                dim_sum = np.sum(coupled_feats_per_dim[dim], axis=0).tolist()
                dim_sum = [f / total_strength for f in dim_sum] if total_strength > 0 else [0.0] * len(sample_coupled_names)
            else:
                dim_sum = [0.0] * len(sample_coupled_names)
            
            for i, val in enumerate(dim_sum):
                features.append(val)
                feat_name = f"{prefix}{dim}_adj_{sample_coupled_names[i]}"
                names.append(feat_name)
                # 匹配特征类别
                if i < len(sample_poly_names):
                    cat = "与相邻细胞的多项式"
                else:
                    cat = "与相邻细胞的其他函数"
                metadata.append((cat, feat_name, f"邻接耦合特征（{sample_coupled_names[i]}）"))

        self.feature_metadata.extend(metadata)
        return features, names

    # ========== 辅助方法 ==========
    def _calc_euclidean_distance(self, coords1, coords2):
        dx = coords1[self.dim_indices['x']] - coords2[self.dim_indices['x']]
        dy = coords1[self.dim_indices['y']] - coords2[self.dim_indices['y']]
        dz = coords1[self.dim_indices['z']] - coords2[self.dim_indices['z']]
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def _process_adjacency_matrix(self, cell_name: str, json_adj: Dict[str, float], current_cell_names: List[str]) -> Dict[str, float]:
        valid_adj = {}
        for neighbor, strength in json_adj.items():
            if neighbor not in current_cell_names or neighbor == cell_name:
                continue
            normalized_strength = max(0.0, min(1.0, float(strength)))
            valid_adj[neighbor] = normalized_strength
            if not (0.0 <= float(strength) <= 1.0):
                self.debug_stats['invalid_adj_strength'] += 1
        
        if len(valid_adj) == 0:
            self.debug_stats['empty_adj_matrix'] += 1
        self.debug_stats['processed_cells'] += 1
        return valid_adj

    # ========== 数据处理主函数 ==========
    def process_json_data(self, json_data: Dict) -> pd.DataFrame:
        self.feature_metadata = []
        all_rows = []
        all_feature_names = None
        self.debug_stats['total_groups'] = len(json_data)

        if not json_data:
            return None

        for group_idx_str, group_data in json_data.items():
            group_idx = int(group_idx_str)
            timepoint_cells = {}
            
            for t_str, time_data in group_data.items():
                t = int(t_str)
                cells = {}
                cell_names = list(time_data.keys())
                
                for cell_name, cell_info in time_data.items():
                    adj_matrix = self._process_adjacency_matrix(
                        cell_name=cell_name,
                        json_adj=cell_info.get('adjacency_matrix', {}),
                        current_cell_names=cell_names
                    )
                    
                    cells[cell_name] = {
                        'self_coords': np.array(cell_info['self_coords']),
                        'adjacency_matrix': adj_matrix,
                        'gi_group': cell_info['gi_group']
                    }
                timepoint_cells[t] = cells

            # 处理母-子组合
            for combo in self.mother_daughter_combinations:
                mother_name, daughter_name = combo.split('-')
                mother_info = self.cell_relations[mother_name]
                mother_gi, daughter_gi = mother_info['mother_gi'], mother_info['daughter_gi']
                mother_t, daughter_t = self.gi_to_t.get(mother_gi, -1), self.gi_to_t.get(daughter_gi, -1)

                if mother_t == -1 or daughter_t == -1:
                    continue
                if mother_t not in timepoint_cells or daughter_t not in timepoint_cells:
                    continue
                mother_cells = timepoint_cells[mother_t]
                daughter_cells = timepoint_cells[daughter_t]

                # 校验母/子细胞
                if mother_name not in mother_cells:
                    self.debug_stats['missing_mother'] += 1
                    continue
                if mother_cells[mother_name]['gi_group'] != mother_gi:
                    self.debug_stats['wrong_gi_mother'] += 1
                    continue
                if daughter_name not in daughter_cells:
                    self.debug_stats['missing_daughter'] += 1
                    continue
                if daughter_cells[daughter_name]['gi_group'] != daughter_gi:
                    self.debug_stats['wrong_gi_daughter'] += 1
                    continue

                self.debug_stats['valid_combinations_found'] += 1

                # 提取母细胞数据并生成特征
                mother_data = mother_cells[mother_name]
                mother_coords = mother_data['self_coords']
                mother_adj_matrix = mother_data['adjacency_matrix']

                # 特征组合（按新分类整合）
                # 1. 细胞自身的单维度
                self_single_feats, self_single_names = self._self_single_dim_features(mother_coords, prefix='mother_')
                # 2. 细胞自身的耦合多项式
                self_coupled_poly_feats, self_coupled_poly_names = self._self_coupled_polynomial_features(mother_coords, prefix='mother_')
                # 3. 细胞自身的其他函数
                self_other_feats, self_other_names = self._self_other_functions_features(mother_coords, prefix='mother_')
                # 4. 与相邻细胞的特征（多项式+其他函数）
                adj_feats, adj_names = self._sum_with_adj_cells(mother_name, mother_coords, mother_cells, mother_adj_matrix, prefix='mother_')

                # 整合所有特征
                combo_features = (self_single_feats + self_coupled_poly_feats + self_other_feats + adj_feats)
                combo_names = (self_single_names + self_coupled_poly_names + self_other_names + adj_names)

                # 初始化特征列名
                if all_feature_names is None:
                    all_feature_names = [
                        'group_idx', 'mother_gi', 'daughter_gi',
                        'mother_t', 'daughter_t', 'cell_combination'
                    ] + combo_names

                # 构造行数据
                row_data = [group_idx, mother_gi, daughter_gi, mother_t, daughter_t, combo] + combo_features
                all_rows.append(pd.Series(row_data, index=all_feature_names))

        # 过滤偶数行
        filtered_rows = [row for i, row in enumerate(all_rows) if i % 2 == 0]
        self.debug_stats['valid_combinations_found'] = len(filtered_rows)

        if len(filtered_rows) == 0:
            pass
        return pd.DataFrame(filtered_rows) if filtered_rows else None

    # ========== 导出特征表格（Markdown格式） ==========
    def export_feature_table(self, output_path: str = "cell_feature_table.md") -> None:
        if not self.feature_metadata:
            return

        # 去重 + 按开关过滤
        unique_metadata = []
        seen_names = set()
        for meta in self.feature_metadata:
            feat_class, feat_name, formula = meta
            # 根据开关过滤特征
            if (feat_class == "细胞自身的单维度" and not self.feature_toggles['self_single_dim']) or \
               (feat_class == "细胞自身的耦合多项式" and not self.feature_toggles['self_coupled_polynomial']) or \
               (feat_class == "与相邻细胞的多项式" and not self.feature_toggles['adjacent_polynomial']) or \
               (feat_class == "细胞自身的其他函数" and not self.feature_toggles['self_other_functions']) or \
               (feat_class == "与相邻细胞的其他函数" and not self.feature_toggles['adjacent_other_functions']):
                continue
            if feat_name not in seen_names:
                seen_names.add(feat_name)
                unique_metadata.append(meta)

        # 按特征类别排序
        unique_metadata.sort(key=lambda x: x[0])

        # 生成Markdown表格
        md_content = "# 细胞特征提取器 - 特征清单（带分类开关）\n\n"
        md_content += "## 特征分类开关状态\n"
        for cat, status in self.feature_toggles.items():
            md_content += f"- {cat}: {'开启' if status else '关闭'}\n"
        md_content += "\n## 特征详情\n\n"
        md_content += "| 特征类别               | 特征名称                                   | 计算公式                                                                 |\n"
        md_content += "|------------------------|--------------------------------------------|--------------------------------------------------------------------------|\n"
        for feat_class, feat_name, formula in unique_metadata:
            md_content += f"| {feat_class} | {feat_name} | {formula} |\n"

        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

    # ========== 数据加载与保存 ==========
    def load_and_process(self, json_file_path: str = "all_222groups_with_adjacency.json") -> pd.DataFrame:
        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            return None

        # 处理数据（生成特征元数据）
        feature_table = self.process_json_data(json_data)

        if feature_table is not None and not feature_table.empty:
            # 保存特征数据CSV
            output_csv = "cell_feature_data_with_3d_and_toggles.csv"
            feature_table.to_csv(output_csv, index=False)
            
            # 导出特征表格Markdown
            self.export_feature_table()
            
            return feature_table
        else:
            return None

def build_feature_table_from_events(
    events_df: pd.DataFrame,
    timepoint_cells_by_group: Dict[int, Dict[int, Dict[str, Dict]]],
    extractor: CellFeatureExtractor,
) -> pd.DataFrame:
    """
    使用 notebook 里的特征构造逻辑，但改成“每个分裂事件一行”。
    不再按 mother-daughter 单个组合生成两行，也不再做偶数行过滤。
    """
    rows = []
    for _, event in events_df.iterrows():
        group_idx = int(event["group_idx"])
        mother_t = int(event["mother_t"])
        mother_name = event["mother_name"]
        transition = event["transition"]

        timepoint_cells = timepoint_cells_by_group[group_idx]
        mother_cells = timepoint_cells[mother_t]
        mother_data = mother_cells[mother_name]
        mother_coords = mother_data["self_coords"]
        mother_adj_matrix = mother_data["adjacency_matrix"]

        extractor.feature_metadata = []
        self_single_feats, self_single_names = extractor._self_single_dim_features(mother_coords, prefix="mother_")
        self_coupled_poly_feats, self_coupled_poly_names = extractor._self_coupled_polynomial_features(mother_coords, prefix="mother_")
        self_other_feats, self_other_names = extractor._self_other_functions_features(mother_coords, prefix="mother_")
        adj_feats, adj_names = extractor._sum_with_adj_cells(
            mother_name,
            mother_coords,
            mother_cells,
            mother_adj_matrix,
            prefix="mother_",
        )

        combo_features = self_single_feats + self_coupled_poly_feats + self_other_feats + adj_feats
        combo_names = self_single_names + self_coupled_poly_names + self_other_names + adj_names

        row = {
            "sample_id": event["sample_id"],
            "group_idx": group_idx,
            "transition": transition,
            "mother_t": int(event["mother_t"]),
            "daughter_t": int(event["daughter_t"]),
            "mother_name": event["mother_name"],
            "daughter1_name": event["daughter1_name"],
            "daughter2_name": event["daughter2_name"],
        }
        row.update(dict(zip(combo_names, combo_features)))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["sample_id"])
    return pd.DataFrame(rows)
