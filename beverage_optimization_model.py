"""
饮料生产企业线性规划优化模型
运筹学专家系统 - 解决原料和运输双重约束下的利润最大化问题
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
# 注：模型核心功能与 Streamlit、Plotly 等可视化库解耦。
# 这些依赖在模型求解过程中并不会用到，但如果在其他模块中需要可视化时可单独引入。
# 为避免在无可视化环境下运行测试时引发 ImportError，这里将其设为可选导入。
try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # 在模型逻辑中不会用到

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.express as px  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except ImportError:
    go = px = make_subplots = None  # 可视化仅在 Streamlit 应用中使用
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BeverageOptimizationModel:
    """
    饮料生产企业线性规划优化模型类
    
    该类构建了一个完整的线性规划模型，用于解决饮料生产企业在原料供应和运输能力
    双重约束条件下的利润最大化问题。
    """
    
    def __init__(self):
        """初始化模型参数"""
        # 定义饮料种类和相关参数
        self.beverage_types = ['碳酸饮料', '果汁饮料', '茶饮料', '功能饮料', '矿泉水']
        self.n_beverages = len(self.beverage_types)
        
        # 定义原料种类
        self.material_types = ['白砂糖', '浓缩果汁', '茶叶提取物', '功能成分', '包装材料']
        self.n_materials = len(self.material_types)
        
        # 定义运输区域
        self.transport_regions = ['华北区', '华东区', '华南区', '西南区', '西北区']
        self.n_regions = len(self.transport_regions)
        
        # 初始化默认参数
        self.setup_default_parameters()
        
    def setup_default_parameters(self):
        """设置默认模型参数"""
        
        # 1. 利润参数 (元/升)
        self.profits = np.array([8.5, 12.0, 10.5, 15.0, 6.0])  # 各饮料单位利润
        
        # 2. 原料消耗矩阵 (单位: 千克/升)
        # 每行代表一种原料，每列代表一种饮料
        self.material_consumption = np.array([
            [0.15, 0.08, 0.06, 0.10, 0.02],  # 白砂糖
            [0.02, 0.25, 0.03, 0.05, 0.01],  # 浓缩果汁
            [0.01, 0.02, 0.20, 0.08, 0.01],  # 茶叶提取物
            [0.00, 0.00, 0.00, 0.15, 0.00],  # 功能成分
            [0.10, 0.12, 0.11, 0.14, 0.08]   # 包装材料
        ])
        
        # 3. 原料供应限制 (千克)
        self.material_limits = np.array([15000, 8000, 6000, 2000, 12000])
        
        # 4. 运输能力限制 (升)
        self.transport_limits = np.array([3000, 2500, 2000, 1800, 1200])
        
        # 5. 各饮料在各区域的需求量权重
        self.demand_weights = np.array([
            [0.25, 0.30, 0.20, 0.15, 0.10],  # 碳酸饮料
            [0.20, 0.35, 0.25, 0.15, 0.05],  # 果汁饮料
            [0.30, 0.25, 0.20, 0.20, 0.05],  # 茶饮料
            [0.35, 0.30, 0.20, 0.10, 0.05],  # 功能饮料
            [0.15, 0.25, 0.30, 0.20, 0.10]   # 矿泉水
        ])
        
        # 6. 上期销售情况 (升)
        self.previous_sales = np.array([2000, 1500, 1200, 800, 2500])
        
        # 7. 最小生产量要求 (销售量的80%)
        self.min_production = 0.8 * self.previous_sales
        
        # 8. 最大生产能力限制
        self.max_production_multiplier = 1.5
        
    def build_matrices(self):
        """
        构建线性规划的标准形式矩阵
        
        目标函数: max c^T x
        约束条件: Ax <= b
        """
        
        # 决策变量: x = [x1, x2, x3, x4, x5] 各饮料生产量
        
        # 目标函数系数 (最大化问题需要转换为最小化)
        c = -self.profits  # 负号因为linprog默认最小化
        
        # 约束矩阵 A 和约束向量 b
        constraint_list = []
        constraint_rhs = []
        
        # 1. 原料约束
        for i in range(self.n_materials):
            constraint_list.append(self.material_consumption[i, :])
            constraint_rhs.append(self.material_limits[i])
        
        # 2. 运输能力约束
        # 总运输量不能超过各区域的运输能力
        for j in range(self.n_regions):
            # 计算各饮料在该区域的运输量
            transport_constraint = np.zeros(self.n_beverages)
            for i in range(self.n_beverages):
                # 假设生产量按需求权重分配到各区域
                transport_constraint[i] = self.demand_weights[i, j]
            constraint_list.append(transport_constraint)
            constraint_rhs.append(self.transport_limits[j])
        
        # 3. 最小生产量约束
        for i in range(self.n_beverages):
            min_constraint = np.zeros(self.n_beverages)
            min_constraint[i] = -1  # -x_i <= -min_production_i
            constraint_list.append(min_constraint)
            constraint_rhs.append(-self.min_production[i])
        
        # 4. 最大生产能力约束
        for i in range(self.n_beverages):
            max_constraint = np.zeros(self.n_beverages)
            max_constraint[i] = 1
            constraint_list.append(max_constraint)
            constraint_rhs.append(self.max_production_multiplier * self.previous_sales[i])
        
        A = np.array(constraint_list)
        b = np.array(constraint_rhs)
        
        return c, A, b
    
    def solve_model(self):
        """
        使用单纯形法求解线性规划模型
        
        Returns:
            result: 包含求解结果的字典
        """
        try:
            # 构建模型矩阵
            c, A, b = self.build_matrices()
            
            # 求解线性规划问题
            # method='highs' 使用HiGHS求解器，比单纯形法更高效
            result = linprog(
                c=c,
                A_ub=A,
                b_ub=b,
                method='highs',
                options={'disp': False}
            )
            
            if result.success:
                # 提取求解结果
                solution = {
                    'status': '最优解找到',
                    'optimal_value': -result.fun,  # 转换回最大化问题
                    'decision_variables': result.x,
                    'shadow_prices': result.ineqlin.marginals if hasattr(result, 'ineqlin') else None,
                    'reduced_costs': result.reduced_costs if hasattr(result, 'reduced_costs') else None,
                    'slack_variables': result.slack,
                    'iterations': result.nit,
                    'success': True
                }
                
                # 计算各约束的影子价格
                solution['constraint_analysis'] = self.analyze_constraints(result, A, b)
                
                return solution
            else:
                return {
                    'status': f'求解失败: {result.message}',
                    'success': False,
                    'message': result.message
                }
                
        except Exception as e:
            return {
                'status': f'求解过程出错: {str(e)}',
                'success': False,
                'message': str(e)
            }
    
    def analyze_constraints(self, result, A, b):
        """
        分析约束条件，计算影子价格和松弛变量
        
        Args:
            result: linprog求解结果
            A: 约束矩阵
            b: 约束右侧向量
            
        Returns:
            dict: 约束分析结果
        """
        analysis = {
            'material_constraints': {},
            'transport_constraints': {},
            'production_constraints': {},
            'binding_constraints': [],
            'non_binding_constraints': []
        }
        
        # 获取松弛变量
        slack = result.slack if hasattr(result, 'slack') else np.zeros(len(b))
        
        # 分析各类约束
        constraint_index = 0
        
        # 1. 原料约束分析
        for i in range(self.n_materials):
            is_binding = slack[constraint_index] < 1e-6
            analysis['material_constraints'][self.material_types[i]] = {
                'usage': np.dot(A[constraint_index], result.x),
                'limit': b[constraint_index],
                'slack': slack[constraint_index],
                'utilization_rate': np.dot(A[constraint_index], result.x) / b[constraint_index],
                'is_binding': is_binding,
                'shadow_price': result.ineqlin.marginals[constraint_index] if hasattr(result, 'ineqlin') and result.ineqlin.marginals is not None else 0
            }
            
            if is_binding:
                analysis['binding_constraints'].append(f"原料约束 - {self.material_types[i]}")
            else:
                analysis['non_binding_constraints'].append(f"原料约束 - {self.material_types[i]}")
            
            constraint_index += 1
        
        # 2. 运输约束分析
        for j in range(self.n_regions):
            is_binding = slack[constraint_index] < 1e-6
            analysis['transport_constraints'][self.transport_regions[j]] = {
                'usage': np.dot(A[constraint_index], result.x),
                'limit': b[constraint_index],
                'slack': slack[constraint_index],
                'utilization_rate': np.dot(A[constraint_index], result.x) / b[constraint_index],
                'is_binding': is_binding,
                'shadow_price': result.ineqlin.marginals[constraint_index] if hasattr(result, 'ineqlin') and result.ineqlin.marginals is not None else 0
            }
            
            if is_binding:
                analysis['binding_constraints'].append(f"运输约束 - {self.transport_regions[j]}")
            else:
                analysis['non_binding_constraints'].append(f"运输约束 - {self.transport_regions[j]}")
            
            constraint_index += 1
        
        # 3. 生产约束分析（最小和最大）
        for i in range(self.n_beverages):
            # 最小生产约束
            min_is_binding = slack[constraint_index] < 1e-6
            analysis['production_constraints'][f"{self.beverage_types[i]}_min"] = {
                'production': result.x[i],
                'minimum': self.min_production[i],
                'slack': slack[constraint_index],
                'is_binding': min_is_binding
            }
            constraint_index += 1
            
            # 最大生产约束
            max_is_binding = slack[constraint_index] < 1e-6
            analysis['production_constraints'][f"{self.beverage_types[i]}_max"] = {
                'production': result.x[i],
                'maximum': self.max_production_multiplier * self.previous_sales[i],
                'slack': slack[constraint_index],
                'is_binding': max_is_binding
            }
            constraint_index += 1
        
        return analysis
    
    def sensitivity_analysis(self, solution):
        """
        进行灵敏度分析
        
        Args:
            solution: 求解结果
            
        Returns:
            dict: 灵敏度分析结果
        """
        if not solution['success']:
            return {'error': '无法对无解模型进行灵敏度分析'}
        
        analysis = {
            'objective_coefficients': {},
            'rhs_changes': {},
            'recommendations': []
        }
        
        # 分析目标函数系数的允许变化范围
        base_profits = self.profits.copy()
        base_solution = solution['decision_variables']
        tol = 1e-3
        max_iter_profit = 10
        for i, beverage in enumerate(self.beverage_types):
            # 初始化信息
            info = {
                'current_profit': base_profits[i],
                'optimal_production': base_solution[i],
                'reduced_cost': solution['reduced_costs'][i] if solution['reduced_costs'] is not None else 0
            }
            # 如果减少成本为正，说明该变量在最优解中为0
            if solution['reduced_costs'] is not None and solution['reduced_costs'][i] > 1e-6:
                analysis['recommendations'].append(
                    f"{beverage}的当前利润过低，建议提高利润至少{solution['reduced_costs'][i]:.2f}元/升或停止生产"
                )

            # 计算允许的利润变化范围，使最优解不发生变化
            orig_profit = base_profits[i]
            step = max(abs(orig_profit) * 0.1, 0.5)  # 使用利润的10%或0.5元作为步长
            min_profit = orig_profit
            max_profit = orig_profit

            # 扫描利润增加
            for k in range(1, max_iter_profit + 1):
                new_profit = orig_profit + step * k
                # 构建新模型以测试利润变化
                test_model = BeverageOptimizationModel()
                # 复制现有参数
                test_model.profits = base_profits.copy()
                test_model.profits[i] = new_profit
                test_model.material_limits = self.material_limits.copy()
                test_model.transport_limits = self.transport_limits.copy()
                test_model.demand_weights = self.demand_weights.copy()
                test_model.previous_sales = self.previous_sales.copy()
                test_model.min_production = self.min_production.copy()
                test_model.max_production_multiplier = self.max_production_multiplier
                # 求解
                sol = test_model.solve_model()
                if sol['success'] and np.allclose(sol['decision_variables'], base_solution, atol=tol, rtol=0):
                    max_profit = new_profit
                else:
                    break

            # 扫描利润减少
            for k in range(1, max_iter_profit + 1):
                new_profit = orig_profit - step * k
                if new_profit < 0:
                    break
                test_model = BeverageOptimizationModel()
                test_model.profits = base_profits.copy()
                test_model.profits[i] = new_profit
                test_model.material_limits = self.material_limits.copy()
                test_model.transport_limits = self.transport_limits.copy()
                test_model.demand_weights = self.demand_weights.copy()
                test_model.previous_sales = self.previous_sales.copy()
                test_model.min_production = self.min_production.copy()
                test_model.max_production_multiplier = self.max_production_multiplier
                sol = test_model.solve_model()
                if sol['success'] and np.allclose(sol['decision_variables'], base_solution, atol=tol, rtol=0):
                    min_profit = new_profit
                else:
                    break

            info['range'] = (min_profit, max_profit)
            analysis['objective_coefficients'][beverage] = info

        # 分析约束右侧的变化影响
        constraint_analysis = solution['constraint_analysis']

        # 原料约束分析
        for idx, material in enumerate(self.material_types):
            if material in constraint_analysis['material_constraints']:
                constraint_info = constraint_analysis['material_constraints'][material]
                if constraint_info['is_binding']:
                    # 记录当前限制和影子价格
                    entry = {
                        'current_limit': constraint_info['limit'],
                        'shadow_price': constraint_info['shadow_price'],
                        'recommendation': f"增加{material}供应可提高利润{constraint_info['shadow_price']:.2f}元/千克"
                    }
                    # 计算该原料供应量变化范围
                    base_limit = self.material_limits[idx]
                    step = max(abs(base_limit) * 0.1, 50)  # 以供应量10%或50千克为步长
                    min_limit = base_limit
                    max_limit = base_limit
                    # 扫描增加供应
                    for k in range(1, 6):
                        new_limit = base_limit + step * k
                        test_model = BeverageOptimizationModel()
                        test_model.profits = self.profits.copy()
                        test_model.material_limits = self.material_limits.copy()
                        test_model.material_limits[idx] = new_limit
                        test_model.transport_limits = self.transport_limits.copy()
                        test_model.demand_weights = self.demand_weights.copy()
                        test_model.previous_sales = self.previous_sales.copy()
                        test_model.min_production = self.min_production.copy()
                        test_model.max_production_multiplier = self.max_production_multiplier
                        sol = test_model.solve_model()
                        if sol['success'] and np.allclose(sol['decision_variables'], solution['decision_variables'], atol=1e-3, rtol=0):
                            max_limit = new_limit
                        else:
                            break
                    # 扫描减少供应
                    for k in range(1, 6):
                        new_limit = base_limit - step * k
                        if new_limit <= 0:
                            break
                        test_model = BeverageOptimizationModel()
                        test_model.profits = self.profits.copy()
                        test_model.material_limits = self.material_limits.copy()
                        test_model.material_limits[idx] = new_limit
                        test_model.transport_limits = self.transport_limits.copy()
                        test_model.demand_weights = self.demand_weights.copy()
                        test_model.previous_sales = self.previous_sales.copy()
                        test_model.min_production = self.min_production.copy()
                        test_model.max_production_multiplier = self.max_production_multiplier
                        sol = test_model.solve_model()
                        if sol['success'] and np.allclose(sol['decision_variables'], solution['decision_variables'], atol=1e-3, rtol=0):
                            min_limit = new_limit
                        else:
                            break
                    entry['range'] = (min_limit, max_limit)
                    analysis['rhs_changes'][f'原料_{material}'] = entry

        # 运输约束分析
        for idx, region in enumerate(self.transport_regions):
            if region in constraint_analysis['transport_constraints']:
                constraint_info = constraint_analysis['transport_constraints'][region]
                if constraint_info['is_binding']:
                    entry = {
                        'current_limit': constraint_info['limit'],
                        'shadow_price': constraint_info['shadow_price'],
                        'recommendation': f"增加{region}运输能力可提高利润{constraint_info['shadow_price']:.2f}元/升"
                    }
                    # 计算运输限制的变化范围
                    base_limit = self.transport_limits[idx]
                    step = max(abs(base_limit) * 0.1, 50)  # 以运输能力10%或50升为步长
                    min_limit = base_limit
                    max_limit = base_limit
                    # 扫描增加运输限制
                    for k in range(1, 6):
                        new_limit = base_limit + step * k
                        test_model = BeverageOptimizationModel()
                        test_model.profits = self.profits.copy()
                        test_model.material_limits = self.material_limits.copy()
                        test_model.transport_limits = self.transport_limits.copy()
                        test_model.transport_limits[idx] = new_limit
                        test_model.demand_weights = self.demand_weights.copy()
                        test_model.previous_sales = self.previous_sales.copy()
                        test_model.min_production = self.min_production.copy()
                        test_model.max_production_multiplier = self.max_production_multiplier
                        sol = test_model.solve_model()
                        if sol['success'] and np.allclose(sol['decision_variables'], solution['decision_variables'], atol=1e-3, rtol=0):
                            max_limit = new_limit
                        else:
                            break
                    # 扫描减少运输限制
                    for k in range(1, 6):
                        new_limit = base_limit - step * k
                        if new_limit <= 0:
                            break
                        test_model = BeverageOptimizationModel()
                        test_model.profits = self.profits.copy()
                        test_model.material_limits = self.material_limits.copy()
                        test_model.transport_limits = self.transport_limits.copy()
                        test_model.transport_limits[idx] = new_limit
                        test_model.demand_weights = self.demand_weights.copy()
                        test_model.previous_sales = self.previous_sales.copy()
                        test_model.min_production = self.min_production.copy()
                        test_model.max_production_multiplier = self.max_production_multiplier
                        sol = test_model.solve_model()
                        if sol['success'] and np.allclose(sol['decision_variables'], solution['decision_variables'], atol=1e-3, rtol=0):
                            min_limit = new_limit
                        else:
                            break
                    entry['range'] = (min_limit, max_limit)
                    analysis['rhs_changes'][f'运输_{region}'] = entry
        
        return analysis
    
    def update_parameters(self, params: Dict):
        """
        更新模型参数
        
        Args:
            params: 参数字典
        """
        if 'profits' in params:
            self.profits = np.array(params['profits'])
        
        if 'material_limits' in params:
            self.material_limits = np.array(params['material_limits'])
        
        if 'transport_limits' in params:
            self.transport_limits = np.array(params['transport_limits'])
        
        if 'min_production_ratio' in params:
            ratio = params['min_production_ratio']
            self.min_production = ratio * self.previous_sales
        
        if 'max_production_multiplier' in params:
            self.max_production_multiplier = params['max_production_multiplier']

# 创建全局模型实例
model = BeverageOptimizationModel()