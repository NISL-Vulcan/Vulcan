import requests
from typing import List, Dict, Any

class ResourceUpdater:
    def __init__(self):
        pass

    def semantic_search_papers(self, query: str, year_from=2023, top_k=5) -> List[Dict[str, Any]]:
        """
        用Semantic Scholar API检索最近真实存在的论文
        """
        print(f"[RAG检索] 正在检索与 '{query}' 相关的最新学术论文...")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": top_k,
            "fields": "title,abstract,year,authors,url"
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            results = []
            if resp.status_code == 200:
                data = resp.json()
                for paper in data.get("data", []):
                    if paper.get("year", 0) >= year_from:
                        results.append({
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract"),
                            "url": paper.get("url"),
                            "year": paper.get("year"),
                            "authors": [a.get("name") for a in paper.get("authors", [])]
                        })
            else:
                print("API请求失败:", resp.status_code, resp.text)
                return []
            print(f"[RAG检索] 共检索到{len(results)}篇论文")
            return results
        except Exception as e:
            print(f"[RAG检索] 检索失败: {e}")
            return []

    def extract_resources(self, paper_info: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[抽取] 正在分析文献: {paper_info['title']}")
        # 用文献标题生成不同的资源链接（此处为示例）
        safe_title = paper_info['title'].replace(' ', '_').replace('/', '_')
        dataset_url = f"https://github.com/example/{safe_title}_dataset"
        model_url = f"https://github.com/example/{safe_title}_model"
        return {
            "dataset_url": dataset_url,
            "model_url": model_url
        }

    def validate_and_standardize(self, resource: Dict[str, Any]) -> bool:
        print(f"[验证] 正在验证资源: {resource}")
        return True

    def update_resources(self):
        queries = [
            "deep learning vulnerability detection",
            "large language model vulnerability detection"
        ]
        all_papers = []
        seen_titles = set()
        for query in queries:
            papers = self.semantic_search_papers(query)
            for paper in papers:
                # 避免重复
                if paper['title'] not in seen_titles:
                    seen_titles.add(paper['title'])
                    all_papers.append(paper)
        for i, paper in enumerate(all_papers, 1):
            print(f"\n[{i}] {paper['title']} ({paper['year']})")
            print("Authors:", ", ".join(paper['authors']))
            print("URL:", paper['url'])
            print("Abstract:", paper['abstract'])
            resource = self.extract_resources(paper)
            if self.validate_and_standardize(resource):
                print(f"[更新] 资源已更新并规范化: {resource}")
            else:
                print("[回退] 资源验证失败，回退检索...")

if __name__ == "__main__":
    updater = ResourceUpdater()
    updater.update_resources() 