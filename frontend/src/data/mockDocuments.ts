/**
 * @deprecated No longer used in UI. The Documents page and Dashboard now fetch
 * from GET /api/documents. Kept only for reference.
 */
import type { ApiDocument } from '@/types/document';

export const mockDocuments: ApiDocument[] = [
  {
    paper_id: 'PEXAMPLE1',
    filename: 'Attention Is All You Need.pdf',
    status: 'completed',
  },
  {
    paper_id: 'PEXAMPLE2',
    filename: 'BERT: Pre-training of Deep Bidirectional Transformers.pdf',
    status: 'completed',
  },
  {
    paper_id: 'PEXAMPLE3',
    filename: 'GPT-4 Technical Report.pdf',
    status: 'processing',
  },
  {
    paper_id: 'PEXAMPLE4',
    filename: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf',
    status: 'completed',
  },
  {
    paper_id: 'PEXAMPLE5',
    filename: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.pdf',
    status: 'failed',
  },
];
