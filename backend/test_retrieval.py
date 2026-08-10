
import sys
sys.stdout.reconfigure(encoding='utf-8')
from rag import retrieve_relevant_documents
docs = retrieve_relevant_documents('?????????', top_k=5)
print('DOCS FOR MANOHARAPUL:')
for i, d in enumerate(docs):
    print(f'---\nDoc {i+1}: ' + d['page_content'])

docs2 = retrieve_relevant_documents('???????? ???', top_k=5)
print('\nDOCS FOR SUBHAKAMANA CHOWK:')
for i, d in enumerate(docs2):
    print(f'---\nDoc {i+1}: ' + d['page_content'])

