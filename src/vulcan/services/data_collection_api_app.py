#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENAPIEN
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

# EN
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EN
jobs = {}

class DataCollectionJob:
    """EN"""
    def __init__(self, job_id: str, collection_type: str = "comprehensive"):
        self.job_id = job_id
        self.collection_type = collection_type
        self.status = "pending"  # pending, running, completed, failed
        self.status_description = "EN"
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.collection_results = {}
        self.error = None
    
    def add_log(self, message: str):
        """EN"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # EN
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]
    
    def get_logs(self, limit: int = None) -> list:
        """EN"""
        if limit:
            return self.logs[-limit:]
        return self.logs
    
    def get_recent_logs(self, count: int = 50) -> list:
        """EN"""
        return self.logs[-count:]
    
    def get_duration(self) -> str:
        """EN"""
        if not self.start_time:
            return "EN"
        if not self.end_time:
            end_time = datetime.now()
        else:
            end_time = self.end_time
        
        duration = end_time - self.start_time
        return str(duration).split('.')[0]  # EN

def run_data_collection(job: DataCollectionJob):
    """EN"""
    try:
        job.status = "running"
        job.status_description = "EN"
        job.start_time = datetime.now()
        job.add_log(" EN")
        
        job.add_log(" EN...")
        collector = DataCollector(job.collection_type)
        
        job.add_log(" EN...")
        
        # EN
        result = collector.collect_sample_data()
        
        if result["success"]:
            job.status = "completed"
            job.status_description = "EN"
            job.collection_results = result
            job.add_log(" EN")
            job.add_log(f" EN: {result['stats']}")
            job.add_log(f" EN: {result['output_file']}")
        else:
            job.status = "failed"
            job.status_description = "EN"
            job.error = "EN"
            job.add_log(" EN")
            
    except Exception as e:
        job.status = "failed"
        job.status_description = f"EN: {str(e)}"
        job.error = str(e)
        job.add_log(f" EN: {str(e)}")
        logger.error(f"EN: {e}")
        traceback.print_exc()
    
    finally:
        job.end_time = datetime.now()
        if job.start_time:
            job.duration = (job.end_time - job.start_time).total_seconds()

@app.route('/api/start-data-collection', methods=['POST'])
def start_data_collection():
    """EN"""
    try:
        data = request.get_json()
        collection_type = data.get('collection_type', 'comprehensive')
        
        # ENID
        job_id = str(uuid.uuid4())
        
        # EN
        job = DataCollectionJob(job_id, collection_type)
        jobs[job_id] = job
        
        # EN
        thread = threading.Thread(target=run_data_collection, args=(job,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"EN: {job_id}")
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "message": f"EN,ENID: {job_id}",
            "collection_type": collection_type
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-status/<job_id>', methods=['GET'])
def get_data_collection_status(job_id: str):
    """EN"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        # EN
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
        
        # EN,EN
        if job.status == "completed" and job.collection_results:
            response_data["collection_results"] = job.collection_results
        
        # EN,EN
        if job.status == "failed" and job.error:
            response_data["error"] = job.error
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-collection-logs/<job_id>', methods=['GET'])
def get_data_collection_logs(job_id: str):
    """EN"""
    try:
        if job_id not in jobs:
            return jsonify({
                "success": False,
                "error": "EN"
            }), 404
        
        job = jobs[job_id]
        
        # EN
        limit = request.args.get('limit', type=int)
        
        logs = job.get_logs(limit)
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "logs": logs,
            "total_logs": len(job.logs)
        })
        
    except Exception as e:
        logger.error(f"EN: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """EN"""
    return jsonify({
        "status": "healthy",
        "service": "data-collection-api",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print(" ENAPIEN...")
    print(" ENAPIEN:")
    print("  POST /api/start-data-collection     - EN")
    print("  GET  /api/data-collection-status/<job_id> - EN")
    print("  GET  /api/data-collection-logs/<job_id>   - EN")
    print("  GET  /api/health                    - EN")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False) 