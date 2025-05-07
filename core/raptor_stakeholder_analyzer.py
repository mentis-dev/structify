#!/usr/bin/env python3
import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import logging

from core.stakeholder_analyzer import StakeholderAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RaptorStakeholderAnalyzer")


class RaptorStakeholderAnalyzer:
    """
    Analyzes stakeholders, factors, and pain points from RAPTOR clusters.
    Combines RAPTOR's document clustering with StakeholderAnalyzer's extraction capabilities.
    Retrieves all level 0 documents that are descendants of selected clusters.
    """

    def __init__(
            self,
            raptor_service,
            model: str = "openai/gpt-4o",
            output_dir: str = "raptor_extraction",
            namespace: str = "default",
            verbose: bool = True
    ):
        """
        Initialize the RAPTOR Stakeholder Analyzer.

        Args:
            raptor_service: Instance of RaptorService for document retrieval
            model: The LLM model to use for extraction
            output_dir: Directory to save extraction results
            namespace: Namespace for RAPTOR retrieval
            verbose: Whether to show detailed logs
        """
        self.raptor_service = raptor_service
        self.namespace = namespace
        self.verbose = verbose

        # Initialize the StakeholderAnalyzer (remains unchanged)
        self.stakeholder_analyzer = StakeholderAnalyzer(
            model=model,
            output_dir=output_dir
        )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    async def get_clusters_at_level(
            self,
            level: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Retrieve document clusters at the specified hierarchy level from RAPTOR.

        Args:
            level: The hierarchy level to get clusters from (default is 1)

        Returns:
            List of cluster metadata with their IDs
        """
        if self.verbose:
            logger.info(f"Retrieving clusters at level {level} from namespace '{self.namespace}'")

        try:
            # Query documents at the specified level
            level_docs = self.raptor_service.manager.query_by_metadata(
                metadata_filter={"level": level},
                namespace=self.namespace
            )

            if not level_docs:
                logger.warning(f"No documents found at level {level} in namespace '{self.namespace}'")
                return []

            if self.verbose:
                logger.info(f"Found {len(level_docs)} clusters at level {level}")

            # Format cluster data
            clusters = []
            for i, doc in enumerate(level_docs):
                # Extract relevant metadata
                doc_id = doc.metadata.get("document_id", f"cluster_{i}")
                child_count = doc.metadata.get("child_count", 0)

                clusters.append({
                    "id": doc_id,
                    "level": level,
                    "child_count": child_count,
                    "content_preview": doc.page_content[:100] + "..." if len(
                        doc.page_content) > 100 else doc.page_content,
                    "document": doc
                })

            return clusters

        except Exception as e:
            logger.error(f"Error retrieving level {level} clusters: {str(e)}")
            return []

    async def get_all_descendant_documents(
            self,
            cluster_id: str
    ) -> List[Dict[str, Any]]:
        """
        Recursively retrieve all level 0 documents that are descendants of a cluster.
        This method traverses the hierarchy to find all base documents.

        Args:
            cluster_id: ID of the parent cluster

        Returns:
            List of level 0 documents that are descendants of the cluster
        """
        if self.verbose:
            logger.info(f"Retrieving all descendant documents for cluster {cluster_id}")

        try:
            # Get all documents that have this as an ancestor (recursive approach)
            level0_docs = []
            visited_clusters = set()  # Track clusters we've processed to avoid cycles

            # Start with the immediate children
            await self._recursively_get_descendants(cluster_id, level0_docs, visited_clusters)

            if not level0_docs:
                logger.warning(f"No level 0 documents found for cluster {cluster_id}")
                return []

            if self.verbose:
                logger.info(f"Found {len(level0_docs)} level 0 documents for cluster {cluster_id}")

            return level0_docs

        except Exception as e:
            logger.error(f"Error retrieving descendant documents for cluster {cluster_id}: {str(e)}")
            return []

    async def _recursively_get_descendants(
            self,
            cluster_id: str,
            level0_docs: List[Dict[str, Any]],
            visited_clusters: Set[str]
    ):
        """
        Helper method to recursively traverse the hierarchy and find all level 0 documents.

        Args:
            cluster_id: ID of the current cluster
            level0_docs: List to collect level 0 documents (modified in-place)
            visited_clusters: Set of clusters already visited to prevent cycles
        """
        # Prevent cycles in the hierarchy
        if cluster_id in visited_clusters:
            return

        visited_clusters.add(cluster_id)

        # Query direct children of this cluster
        child_docs = self.raptor_service.manager.query_by_metadata(
            metadata_filter={"parent_id": cluster_id},
            namespace=self.namespace
        )

        if not child_docs:
            return

        for doc in child_docs:
            # Get document level
            doc_level = doc.metadata.get("level", -1)
            doc_id = doc.metadata.get("document_id", "")

            if doc_level == 0:
                # This is a base document - add it to our results
                level0_docs.append({
                    "id": doc_id,
                    "content": doc.page_content,
                    "document": doc
                })
            else:
                # This is an intermediate node - recursively check its children
                await self._recursively_get_descendants(doc_id, level0_docs, visited_clusters)

    async def analyze_cluster(
            self,
            cluster_id: str,
            cluster_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a specific RAPTOR cluster by extracting from all its descendant level 0 documents.

        Args:
            cluster_id: ID of the cluster to analyze
            cluster_name: Optional name for the cluster (for reporting)

        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        cluster_name = cluster_name or f"Cluster {cluster_id}"

        if self.verbose:
            logger.info(f"Analyzing cluster: {cluster_name}")

        try:
            # Get all level 0 descendant documents for this cluster
            level0_docs = await self.get_all_descendant_documents(cluster_id)

            if not level0_docs:
                return {
                    "error": f"No level 0 documents found for cluster {cluster_id}",
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name
                }

            # Combine all content from level 0 documents
            combined_text = ""
            for doc in level0_docs:
                # Add document content with a separator
                combined_text += doc["content"] + "\n\n----------\n\n"

            if self.verbose:
                logger.info(
                    f"Combined {len(level0_docs)} level 0 documents, total length: {len(combined_text)} characters")

            # Use StakeholderAnalyzer to extract from combined text
            doc_id = f"cluster_{cluster_id}"
            result = await self.stakeholder_analyzer.analyze_text(
                text=combined_text,
                doc_id=doc_id,
                doc_name=cluster_name,
                verbose=self.verbose
            )

            # Add cluster metadata to the result
            result["cluster_info"] = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "descendant_count": len(level0_docs),
                "combined_text_length": len(combined_text)
            }

            elapsed_time = time.time() - start_time
            if self.verbose:
                logger.info(f"Completed analysis of cluster {cluster_name} in {elapsed_time:.2f} seconds")

            return result

        except Exception as e:
            error_msg = f"Error analyzing cluster {cluster_id}: {str(e)}"
            logger.error(error_msg)

            return {
                "error": error_msg,
                "cluster_id": cluster_id,
                "cluster_name": cluster_name
            }

    async def analyze_all_clusters(
            self,
            level: int = 1,
            aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze all clusters at a specific level.

        Args:
            level: The hierarchy level to analyze (default is 1)
            aggregate: Whether to aggregate results across all clusters

        Returns:
            Dictionary with individual and optionally aggregated results
        """
        start_time = time.time()

        if self.verbose:
            logger.info(f"Starting analysis of all clusters at level {level}")

        try:
            # Get all clusters at the specified level
            clusters = await self.get_clusters_at_level(level)

            if not clusters:
                return {
                    "error": f"No clusters found at level {level}",
                    "level": level
                }

            # Process each cluster
            results = {}
            for i, cluster in enumerate(clusters):
                cluster_id = cluster["id"]
                cluster_name = f"Level {level} Cluster {i + 1}"

                if self.verbose:
                    logger.info(f"Processing cluster {i + 1}/{len(clusters)}: {cluster_name}")

                result = await self.analyze_cluster(cluster_id, cluster_name)
                results[cluster_id] = result

            # Aggregate results if requested
            if aggregate and results:
                aggregated_results = self._aggregate_cluster_results(results)

                # Save aggregated results
                output_file = os.path.join(
                    self.stakeholder_analyzer.output_dir,
                    f"raptor_level{level}_aggregated_analysis.json"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(aggregated_results, f, indent=2)

                if self.verbose:
                    logger.info(f"Saved aggregated results to {output_file}")

                elapsed_time = time.time() - start_time
                logger.info(f"Completed analysis of {len(clusters)} clusters in {elapsed_time:.2f} seconds")

                return {
                    "individual_results": results,
                    "aggregated_results": aggregated_results,
                    "level": level,
                    "clusters_analyzed": len(clusters),
                    "processing_time": elapsed_time
                }
            else:
                elapsed_time = time.time() - start_time
                logger.info(f"Completed analysis of {len(clusters)} clusters in {elapsed_time:.2f} seconds")

                return {
                    "individual_results": results,
                    "level": level,
                    "clusters_analyzed": len(clusters),
                    "processing_time": elapsed_time
                }

        except Exception as e:
            error_msg = f"Error analyzing clusters at level {level}: {str(e)}"
            logger.error(error_msg)

            return {
                "error": error_msg,
                "level": level
            }

    def _aggregate_cluster_results(
            self,
            cluster_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple clusters into a unified analysis.

        Args:
            cluster_results: Dictionary mapping cluster IDs to their analysis results

        Returns:
            Dictionary with aggregated analysis
        """
        if self.verbose:
            logger.info(f"Aggregating results from {len(cluster_results)} clusters")

        # Prepare input format for StakeholderAnalyzer._aggregate_results
        transformed_results = {}

        for cluster_id, result in cluster_results.items():
            if result.get('error'):
                continue

            # Extract only the necessary fields for aggregation
            transformed_results[cluster_id] = {
                'document_info': {
                    'id': cluster_id,
                    'name': result.get('cluster_info', {}).get('cluster_name', f"Cluster {cluster_id}")
                },
                'stakeholders': result.get('stakeholders', []),
                'factors': result.get('factors', []),
                'pain_points': result.get('pain_points', [])
            }

        # Use the existing aggregation logic from StakeholderAnalyzer
        aggregated_results = self.stakeholder_analyzer._aggregate_results(transformed_results)

        # Add metadata about the clusters
        aggregated_results['source'] = 'raptor_clusters'
        aggregated_results['level'] = next(iter(cluster_results.values())).get('cluster_info', {}).get('level', 1)
        aggregated_results['total_clusters'] = len(cluster_results)

        return aggregated_results

    def get_level_options(self) -> List[int]:
        """
        Get available hierarchy levels in the RAPTOR index.

        Returns:
            List of available levels
        """
        try:
            # Query the manager to get documents at each level
            levels = []

            # Check levels starting from 0 up to a reasonable maximum
            for level in range(10):  # Assume maximum 10 levels
                level_docs = self.raptor_service.manager.query_by_metadata(
                    metadata_filter={"level": level},
                    namespace=self.namespace,
                    limit=1  # We only need to check if any exist
                )

                if level_docs:
                    levels.append(level)

            return levels

        except Exception as e:
            logger.error(f"Error querying available levels: {str(e)}")
            return []


async def analyze_raptor_level(
        raptor_service,
        level: int = 1,
        model: str = "openai/gpt-4o",
        output_dir: str = "raptor_extraction",
        namespace: str = "default",
        verbose: bool = True,
        aggregate: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze all clusters at a specific RAPTOR level.

    Args:
        raptor_service: Instance of RaptorService for document retrieval
        level: The hierarchy level to analyze (default is 1)
        model: The LLM model to use for extraction
        output_dir: Directory to save extraction results
        namespace: Namespace for RAPTOR retrieval
        verbose: Whether to show detailed logs
        aggregate: Whether to aggregate results across all clusters

    Returns:
        Dictionary with analysis results
    """
    analyzer = RaptorStakeholderAnalyzer(
        raptor_service=raptor_service,
        model=model,
        output_dir=output_dir,
        namespace=namespace,
        verbose=verbose
    )

    return await analyzer.analyze_all_clusters(level=level, aggregate=aggregate)


async def analyze_raptor_cluster(
        raptor_service,
        cluster_id: str,
        cluster_name: Optional[str] = None,
        model: str = "openai/gpt-4o",
        output_dir: str = "raptor_extraction",
        namespace: str = "default",
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze a specific RAPTOR cluster.

    Args:
        raptor_service: Instance of RaptorService for document retrieval
        cluster_id: ID of the cluster to analyze
        cluster_name: Optional name for the cluster (for reporting)
        model: The LLM model to use for extraction
        output_dir: Directory to save extraction results
        namespace: Namespace for RAPTOR retrieval
        verbose: Whether to show detailed logs

    Returns:
        Dictionary with analysis results
    """
    analyzer = RaptorStakeholderAnalyzer(
        raptor_service=raptor_service,
        model=model,
        output_dir=output_dir,
        namespace=namespace,
        verbose=verbose
    )

    return await analyzer.analyze_cluster(
        cluster_id=cluster_id,
        cluster_name=cluster_name
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would be imported or passed in a real application
        from services.raptor_service import RaptorService

        # Create RaptorService instance
        raptor_service = RaptorService(
            index_name="example_index",
            tree_depth=3,
            namespace="default"
        )

        # Analyze all clusters at level 1
        results = await analyze_raptor_level(
            raptor_service=raptor_service,
            level=1,
            verbose=True
        )

        print(f"Analyzed {results.get('clusters_analyzed', 0)} clusters")


    asyncio.run(main())