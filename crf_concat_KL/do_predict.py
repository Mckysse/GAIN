import os
from tqdm import tqdm
import pdb
from torch.utils.data import DataLoader

from utils.utils import get_reader, parse_args, get_tagset, load_model
from utils.metric import calculate_macro_f1

if __name__ == '__main__':
    sg = parse_args()
    test_data = get_reader(file_path=sg.test, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model,
                           max_instances=sg.max_instances, max_length=sg.max_length, gazetteer_path=sg.gazetteer)

    model = load_model(sg.model, tag_to_id=get_tagset(sg.iob_tagging))
    if sg.gpus > 0:
        model = model.cuda()

    eval_file = os.path.join(sg.out_dir, 'test_out.txt')

    test_dataloaders = DataLoader(test_data, batch_size=sg.batch_size, collate_fn=model.collate_batch, shuffle=False,
                                  drop_last=False)
    out_str = ''
    for batch in tqdm(test_dataloaders, total=len(test_dataloaders)):
        pred_tags = model.predict_tags(batch)

        for pred_tag_inst in pred_tags:
            out_str += '\n'.join(pred_tag_inst)
            out_str += '\n\n'
    open(eval_file, 'wt').write(out_str)

    macro_f1_score = calculate_macro_f1(sg.test, eval_file)
    score_file = os.path.join(sg.out_dir, 'score_out.txt')
    open(score_file, 'wt').write(str(macro_f1_score))
