from xml.etree import ElementTree

########################################################################################################################
# Parse XML's
########################################################################################################################

docs = ElementTree.parse('skb/ie1_collection.trec')
qrys = ElementTree.parse('skb/ie1_queries.trec')

for node in docs.findall('DOC'):
    record_id = node.find('recordId').text
    record_txt = node.find('text').text
    with open(f'skb/documents/{record_id}', 'w') as f:
        f.write(record_txt)
        print(f'written document {record_id}')

for node in qrys.findall('DOC'):
    record_id = node.find('recordId').text
    record_txt = node.find('text').text
    with open(f'skb/queries/{record_id}', 'w') as f:
        f.write(record_txt)
        print(f'written query {record_id}')
