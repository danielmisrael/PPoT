import ppot, json
import ppot.utils


dataset = ppot.utils.prepare_data("TencentARC/Plot2Code", num_examples=132,
                                      filter_fn = lambda x: "matplotlib" in x["url"], split="test")

file_handle = open("Plot2Code/data/Plot2Code/test/metadata.jsonl", "w")
result = {}
for i in range(132):
    json_line = json.dumps({'code': dataset['code'][i]})
    file_handle.write(json_line+"\n")
file_handle.close()




