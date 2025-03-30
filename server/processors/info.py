import os
from typing import Optional

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from pipecat.pipeline.task import PipelineTask


class InfoProcessor(FrameProcessor):
    """
    Processor to handle information retrieval from knowledge base.
    """
    def __init__(self, llm_context: OpenAILLMContext, context_aggregator, db_path: str = "./chroma_db", knowledge_file: str = "info.md"):
        super().__init__()
        self.llm_context = llm_context
        self.db_path = db_path
        self.knowledge_file = knowledge_file
        self.vector_store: Optional[Chroma] = None
        self._load_knowledge()
        self.task: Optional[PipelineTask] = None
        self.context_aggregator = context_aggregator


        self.llm_context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "getInfo",
                    "description": "Fournit une information spécifique sur les services du cabinet.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"}
                        },
                        "required": ["question"]
                    }
                }
            }
        ])

    def _load_knowledge(self):
        try:
            # Load documents
            loader = TextLoader(self.knowledge_file)
            docs = loader.load()

            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(docs)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            self.vector_store = Chroma.from_documents(split_docs, embeddings, persist_directory=self.db_path)
            logger.info("Knowledge base loaded and indexed.")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def _auto_continue(self):
        if self.task:
            await self.task.queue_frames([self.context_aggregator.user().get_context_frame()])

    def set_task(self, task: PipelineTask):
        self.task = task

    async def handle_get_info(self, function_name, tool_call_id, args, llm, context, result_callback):
        question = args["question"]
        answer = await self._search_knowledge(question)
        self.llm_context.add_message({"role": "system", "content": answer})
        await result_callback(None)
        await self._auto_continue()

    async def _search_knowledge(self, query: str) -> str:
        if not self.vector_store:
            return "Je suis désolée, je n'ai pas accès aux informations en ce moment."

        try:
            results = self.vector_store.similarity_search(query, k=2)
            if not results:
                return "Je n'ai pas trouvé d'information pertinente."

            response = "\n".join([doc.page_content for doc in results])
            return response

        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return "Je suis désolée, je n'ai pas pu récupérer l'information demandée."
