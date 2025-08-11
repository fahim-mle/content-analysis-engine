# ==================================================================================================
# Imports
# ==================================================================================================
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import chromadb
import feedparser
import pymongo
import PyPDF2
import requests

# ==================================================================================================
# Logger Setup
# ==================================================================================================
# Configure logging to display informational messages.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================================================================
# ArxivScraper Class
# ==================================================================================================
class ArxivScraper:
    """
    A class to scrape paper information from ArXiv, download PDFs, extract text,
    and store the data in MongoDB and ChromaDB for further analysis and similarity search.
    """

    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="arxiv_papers"):
        """
        Initializes the ArxivScraper, setting up connections to MongoDB and ChromaDB.

        Args:
            mongo_uri (str): The URI for the MongoDB connection.
            db_name (str): The name of the database to use in MongoDB.
        """
        # ------------------------------------------------------------------------------------------
        # Database Setup
        # ------------------------------------------------------------------------------------------
        # Establish connection to MongoDB and select the database and collection.
        self.mongo_client = pymongo.MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.papers_collection = self.db.papers

        # Initialize a persistent ChromaDB client and create or get a collection for embeddings.
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="arxiv_embeddings", metadata={"description": "ArXiv paper embeddings"}
        )

        # ------------------------------------------------------------------------------------------
        # Indexing and Directories
        # ------------------------------------------------------------------------------------------
        # Create indexes on the MongoDB collection to optimize query performance.
        self.papers_collection.create_index("arxiv_id", unique=True)
        self.papers_collection.create_index("categories")
        self.papers_collection.create_index("published")

        # Create a directory to store downloaded PDF files.
        self.pdf_dir = Path("./arxiv_pdfs")
        self.pdf_dir.mkdir(exist_ok=True)

    # ==============================================================================================
    # ArXiv API and Data Parsing
    # ==============================================================================================
    def build_arxiv_query(
        self,
        categories=None,
        max_results=100,
        start_date=None,
        end_date=None,
    ):
        """
        Constructs the URL for querying the ArXiv API based on specified criteria.

        Args:
            categories (list or str, optional): A list of ArXiv categories to search.
            max_results (int, optional): The maximum number of results to return.
            start_date (str, optional): The start date for the search query (YYYYMMDD).
            end_date (str, optional): The end date for the search query (YYYYMMDD).

        Returns:
            str: The fully constructed ArXiv API query URL.
        """
        base_url = "http://export.arxiv.org/api/query?"

        # Build the search query part of the URL.
        if categories:
            if isinstance(categories, str):
                categories = [categories]
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({cat_query})"
        else:
            search_query = "all"

        # Append date range to the query if provided.
        if start_date:
            search_query += f" AND submittedDate:[{start_date} TO {end_date or '*'}]"

        # Define API parameters.
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        # Combine parameters into a query string.
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{base_url}{query_string}"

    def parse_arxiv_entry(self, entry):
        """
        Parses an individual entry from the ArXiv feed to extract relevant paper data.

        Args:
            entry (feedparser.FeedParserDict): A single paper entry from the ArXiv feed.

        Returns:
            dict: A dictionary containing the parsed paper information.
        """
        # Extract key information from the feed entry.
        arxiv_id = entry.id.split("/abs/")[-1]
        categories = [tag.term for tag in entry.get("tags", [])]
        authors = [author.name for author in entry.get("authors", [])]
        published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
        updated = datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%SZ")
        pdf_url = entry.id.replace("/abs/", "/pdf/") + ".pdf"

        # Assemble the extracted data into a dictionary.
        paper_data = {
            "arxiv_id": arxiv_id,
            "title": entry.title.replace("\n", " ").strip(),
            "summary": entry.summary.replace("\n", " ").strip(),
            "authors": authors,
            "categories": categories,
            "published": published,
            "updated": updated,
            "pdf_url": pdf_url,
            "scraped_at": datetime.utcnow(),
            "pdf_downloaded": False,
            "pdf_processed": False,
        }

        return paper_data

    def fetch_arxiv_papers(
        self,
        categories,
        max_results=1000,
        start_date=None,
        end_date=None,
    ):
        """
        Fetches a list of papers from the ArXiv API based on the search criteria.

        Args:
            categories (list or str): The ArXiv categories to search.
            max_results (int, optional): The maximum number of papers to fetch.
            start_date (str, optional): The start date for the search.
            end_date (str, optional): The end date for the search.

        Returns:
            list: A list of dictionaries, where each dictionary represents a paper.
        """
        url = self.build_arxiv_query(categories, max_results, start_date, end_date)
        logger.info(f"Fetching papers from: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse the XML feed from the response.
            feed = feedparser.parse(response.text)

            # Process each entry in the feed.
            papers = []
            for entry in feed.entries:
                try:
                    paper = self.parse_arxiv_entry(entry)
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Error parsing entry: {e}")
                    continue

            logger.info(f"Successfully parsed {len(papers)} papers")
            return papers

        except requests.RequestException as e:
            logger.error(f"Error fetching from ArXiv: {e}")
            return []

    # ==============================================================================================
    # File Handling
    # ==============================================================================================
    def download_pdf(self, paper_data, timeout=30):
        """
        Downloads the PDF of a paper and saves it to the local directory.

        Args:
            paper_data (dict): The dictionary containing paper information.
            timeout (int, optional): The timeout for the download request.

        Returns:
            str or None: The file path of the downloaded PDF, or None if download fails.
        """
        arxiv_id = paper_data["arxiv_id"]
        pdf_url = paper_data["pdf_url"]
        pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"

        # Check if the PDF has already been downloaded.
        if pdf_path.exists():
            logger.info(f"PDF already exists: {arxiv_id}")
            return str(pdf_path)

        try:
            logger.info(f"Downloading PDF: {arxiv_id}")
            headers = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
            response = requests.get(pdf_url, headers=headers, timeout=timeout)
            response.raise_for_status()

            # Save the PDF content to a file.
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            logger.info(f"PDF downloaded: {arxiv_id}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Error downloading PDF {arxiv_id}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text content from a given PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str or None: The extracted text, or None if extraction fails.
        """
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""

                # Iterate through each page and extract text.
                for page in reader.pages:
                    text += page.extract_text() + "\n"

                return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    # ==============================================================================================
    # Database Operations
    # ==============================================================================================
    def store_paper_mongodb(self, paper_data):
        """
        Stores or updates paper metadata in the MongoDB collection.

        Args:
            paper_data (dict): The dictionary containing paper information.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            # Use 'upsert' to insert a new document or update an existing one.
            result = self.papers_collection.update_one(
                {"arxiv_id": paper_data["arxiv_id"]}, {"$set": paper_data}, upsert=True
            )

            if result.upserted_id:
                logger.info(f"Inserted new paper: {paper_data['arxiv_id']}")
            else:
                logger.info(f"Updated existing paper: {paper_data['arxiv_id']}")

            return True

        except Exception as e:
            logger.error(f"Error storing paper {paper_data['arxiv_id']}: {e}")
            return False

    def store_embedding_chromadb(self, arxiv_id, text, metadata=None):
        """
        Stores the text embedding of a paper in the ChromaDB collection.

        Args:
            arxiv_id (str): The ArXiv ID of the paper.
            text (str): The text content to be embedded.
            metadata (dict, optional): Additional metadata to store with the embedding.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            # Create a unique ID for the document.
            doc_id = f"arxiv_{arxiv_id}"

            # Add the document to ChromaDB, which handles embedding generation.
            self.vector_collection.add(
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata or {"arxiv_id": arxiv_id}],
            )

            logger.info(f"Stored embedding for: {arxiv_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding for {arxiv_id}: {e}")
            return False

    # ==============================================================================================
    # Main Processing Pipeline
    # ==============================================================================================
    def process_paper_pipeline(self, paper_data, download_pdf=True, extract_text=True):
        """
        Executes the complete processing pipeline for a single paper.

        Args:
            paper_data (dict): The dictionary containing paper information.
            download_pdf (bool, optional): Whether to download the PDF.
            extract_text (bool, optional): Whether to extract text from the PDF.

        Returns:
            bool: True if the pipeline completes successfully, False otherwise.
        """
        arxiv_id = paper_data["arxiv_id"]

        # Step 1: Store metadata in MongoDB.
        if not self.store_paper_mongodb(paper_data):
            return False

        # Step 2: Download the PDF if requested.
        pdf_path = None
        if download_pdf:
            pdf_path = self.download_pdf(paper_data)
            if pdf_path:
                # Update the paper's status in MongoDB.
                self.papers_collection.update_one(
                    {"arxiv_id": arxiv_id},
                    {"$set": {"pdf_downloaded": True, "pdf_path": pdf_path}},
                )

        # Step 3: Extract text and store embeddings if requested.
        if extract_text and pdf_path:
            full_text = self.extract_text_from_pdf(pdf_path)
            if full_text:
                # Store the extracted text in MongoDB.
                self.papers_collection.update_one(
                    {"arxiv_id": arxiv_id},
                    {"$set": {"full_text": full_text, "pdf_processed": True}},
                )

                # Prepare text and metadata for embedding.
                embedding_text = (
                    f"{paper_data['title']} {paper_data['summary']} {full_text[:5000]}"
                )
                metadata = {
                    "arxiv_id": arxiv_id,
                    "title": paper_data["title"],
                    "categories": paper_data["categories"],
                    "published": paper_data["published"].isoformat(),
                }

                # Store the embedding in ChromaDB.
                self.store_embedding_chromadb(arxiv_id, embedding_text, metadata)

        return True

    def scrape_and_store(
        self,
        categories,
        max_results=100,
        start_date=None,
        end_date=None,
        download_pdfs=True,
        rate_limit=2,
    ):
        """
        The main function to orchestrate the scraping and storing process.

        Args:
            categories (list or str): The ArXiv categories to scrape.
            max_results (int, optional): The maximum number of papers to process.
            start_date (str, optional): The start date for scraping.
            end_date (str, optional): The end date for scraping.
            download_pdfs (bool, optional): Whether to download PDFs.
            rate_limit (int, optional): The delay in seconds between processing papers.
        """
        logger.info(f"Starting scrape for categories: {categories}")

        # Fetch the list of papers from ArXiv.
        papers = self.fetch_arxiv_papers(categories, max_results, start_date, end_date)

        if not papers:
            logger.warning("No papers found")
            return

        # Process each paper through the pipeline.
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper['arxiv_id']}")

            try:
                self.process_paper_pipeline(
                    paper, download_pdfs, extract_text=download_pdfs
                )

                # Pause to respect API rate limits.
                if rate_limit > 0:
                    time.sleep(rate_limit)

            except Exception as e:
                logger.error(f"Error processing paper {paper['arxiv_id']}: {e}")
                continue

        logger.info(f"Completed scraping {len(papers)} papers")

    # ==============================================================================================
    # Search and Statistics
    # ==============================================================================================
    def search_similar_papers(self, query_text, n_results=10):
        """
        Searches for papers similar to a given query text using vector embeddings.

        Args:
            query_text (str): The text to search for similarity against.
            n_results (int, optional): The number of similar results to return.

        Returns:
            list: A list of papers from MongoDB that are similar to the query.
        """
        try:
            # Query ChromaDB for similar documents.
            results = self.vector_collection.query(
                query_texts=[query_text], n_results=n_results
            )

            # Retrieve full paper details from MongoDB using the IDs from the search results.
            arxiv_ids = [meta["arxiv_id"] for meta in results["metadatas"][0]]
            papers = list(
                self.papers_collection.find(
                    {"arxiv_id": {"$in": arxiv_ids}},
                    {
                        "title": 1,
                        "authors": 1,
                        "categories": 1,
                        "published": 1,
                        "arxiv_id": 1,
                    },
                )
            )

            return papers

        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []

    def get_paper_stats(self):
        """
        Retrieves statistics about the papers stored in the database.

        Returns:
            dict: A dictionary containing various statistics.
        """
        # Count total papers and those with downloaded/processed PDFs.
        total_papers = self.papers_collection.count_documents({})
        pdf_downloaded = self.papers_collection.count_documents(
            {"pdf_downloaded": True}
        )
        pdf_processed = self.papers_collection.count_documents({"pdf_processed": True})

        # Aggregate to find the top 10 most common categories.
        pipeline = [
            {"unwind": "$categories"},
            {"group": {"_id": "$categories", "count": {"$sum": 1}}},
            {"sort": {"count": -1}},
            {"limit": 10},
        ]
        top_categories = list(self.papers_collection.aggregate(pipeline))

        # Compile the statistics into a dictionary.
        stats = {
            "total_papers": total_papers,
            "pdf_downloaded": pdf_downloaded,
            "pdf_processed": pdf_processed,
            "top_categories": top_categories,
        }

        return stats


# ==================================================================================================
# Example Usage
# ==================================================================================================
if __name__ == "__main__":
    """
    Main execution block to demonstrate the functionality of the ArxivScraper.
    This script will:
    1. Initialize the scraper.
    2. Scrape papers from specified AI/ML categories from the last 30 days.
    3. Print statistics about the scraped data.
    4. Perform a similarity search for a sample query.
    """
    # Initialize the scraper.
    scraper = ArxivScraper()

    # Define the categories and date range for scraping.
    categories = [
        "cs.AI",
        "cs.LG",
        "cs.CL",
    ]  # AI, Machine Learning, Computation and Language
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Start the scraping process.
    scraper.scrape_and_store(
        categories=categories,
        max_results=50,  # Using a small number for demonstration purposes.
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        download_pdfs=True,
        rate_limit=2,  # 2-second delay between requests.
    )

    # Retrieve and print statistics.
    stats = scraper.get_paper_stats()
    print(f"Statistics: {stats}")

    # Perform a similarity search and print the results.
    similar_papers = scraper.search_similar_papers(
        "machine learning natural language processing"
    )
    print(f"Found {len(similar_papers)} similar papers")
