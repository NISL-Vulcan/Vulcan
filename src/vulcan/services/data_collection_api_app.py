#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集API端点
"""

import threading
import uuid
from datetime import datetime
import json
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from vulcan.datacollection.data_collector import DataCollector

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储数据收集任务
jobs = {}

class DataCollectionJob:
    """数据收集任务类"""
    def __init__(self, job_id: str, collection_type: str = "comprehensive"):
        self.job_id = job_id
        self.collection_type = collection_type
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "等待启动"
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.collection_results = {}
        self.error = None
    
    def add_log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # 限制日志数量
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]
    
    def get_logs(self, limit: int = None) -> list:
        """获取日志"""
        if limit:
            return self.logs[-limit:]
        return self.logs
    
    def get_recent_logs(self, count: int = 50) -> list:
        """获取最近的日志"""
        return self.logs[-count:]
    
    def get_duration(self) -> str:
        """获取任务持续时间"""
        if not self.start_time:
            return "未开始"
        if not self.end_time:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        
        duration = end_time - self.start_time
        return str(duration).split('.')[0]  # 移除微秒部分

def run_data_collection(job: DataCollectionJob):
    """运行数据收集任务"""
    try:
        job.status = "running"
        job.status_description = "正在收集数据"
        job.start_time = datetime.now()
        job.add_log("🚀 开始数据收集任务")
        
        job.add_log("📦 初始化数据收集器...")
        collector = DataCollector(job.collection_type)
        
        job.add_log("🔍 开始执行数据收集...")
        
        # 执行数据收集
        result = collector.collect_sample_data()
        
        if result["success"]:
            job.status = "completed"
            job.status_description = "数据收集完成"
            job.collection_results = result
            job.add_log("✅ 数据收集任务完成")
            job.add_log(f"📊 收集统计: {result['stats']}")
            job.add_log(f"📁 输出文件: {result['output_file']}")
        else:
            job.status = "failed"
            job.status_description = "数据收集失败"
            job.error = "数据收集执行失败"
            job.add_log("❌ 数据收集任务失败")
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"数据收集出错: {str(e)}"
        job.error = str(e)
        job.add_log(f"❌ 数据收集异常: {str(e)}")
        logger.error(f"数据收集任务异常: {e}")
        traceback.print_exc()
    
    finally:
        job.end_time = datetime.now()
        if job.start_time:
            job.duration = (job.end_time - job.start_time).total_seconds()

@app.route('/api/start-data-collection', methods=['POST'])
def start_data_collection():
    """启动数据收集任务"""
    try:
        data = request.get_json()
        collection_type = data.get('collection_type', 'comprehensive')
        
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 创建数据收集任务
        job = DataCollectionJob(job_id, collection_type)
        jobs[job_id] = job
        
        # 在后台线程中运行数据收集
        thread = threading.Thread(target=run_data_collection, args=(job,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"数据收集任务已启动: {job_id}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"数据收集任务已启动，任务ID: {job_id}",
            "collection_type": collection_type
        })
        
    except Exception as e:
        logger.error(f"启动数据收集失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-status/<job_id>', methods=['GET'])
def get_data_collection_status(job_id: str):
    """获取数据收集状态"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "数据收集任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        # 获取最近的日志
        recent_logs = job.get_recent_logs(20)
        
        response_data = {
            "success": True,
            "job_id": job_id,
            "status": job.status,
            "status_description": job.status_description,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "duration": job.get_duration(),
            "log_count": len(job.logs),
            "recent_logs": recent_logs,
            "collection_type": job.collection_type
        }
        
        # 如果任务完成，添加结果信息
        if job.status == "completed" and job.collection_results:
            response_data["collection_results"] = job.collection_results
        
        # 如果任务失败，添加错误信息
        if job.status == "failed" and job.error:
            response_data["error"] = job.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"获取数据收集状态失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-logs/<job_id>', methods=['GET'])
def get_data_collection_logs(job_id: str):
    """获取数据收集日志"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "数据收集任务不存在"
            }), 404
        
        job = jobs[job_id]
        
        # 获取查询参数
        limit = request.args.get('limit', type=int)
        
        logs = job.get_logs(limit)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "total_logs": len(job.logs)
        })
        
    except Exception as e:
        logger.error(f"获取数据收集日志失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "data-collection-api",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 数据收集API服务启动...")
    print("📋 可用API接口:")
    print("  POST /api/start-data-collection     - 启动数据收集")
    print("  GET  /api/data-collection-status/<job_id> - 获取数据收集状态")
    print("  GET  /api/data-collection-logs/<job_id>   - 获取数据收集日志")
    print("  GET  /api/health                    - 健康检查")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False) 