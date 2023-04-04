import { OpenAI } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import { readFileSync } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";

async function qaBot(query: string) {
  const model = new OpenAI({});
  const source = readFileSync("../assets/state_of_the_union.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([source]);
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({ query });
  console.log(res);
}

qaBot("What did the president say about Justice Breyer?");
