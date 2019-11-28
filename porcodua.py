    /content/PreSumm/src/summarizer2.py	2019-10-29 03:47:19.168619951 +0000
@@ -1,6 +1,6 @@
 #!/usr/bin/env python
 """
+    Main training workflow
 """
 from __future__ import division
 
@@ -10,6 +10,7 @@
 from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs, load_models_abs
 from train_extractive import train_ext, validate_ext, test_ext
 from prepro import data_builder
 import glob, os
 
 model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
                'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']
@@ -25,7 +26,7 @@
 
 
 
 def init_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
     parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
@@ -127,6 +128,10 @@
     device = "cpu" if args.visible_gpus == '-1' else "cuda"
     device_id = 0 if device == "cuda" else -1
 
     return args, device_id
 
 if __name__ == '__main__':
     args, device_id = init_args()
     print(args.task, args.mode) 
 
     cp = args.test_from
@@ -137,28 +142,12 @@
 
     predictor = load_models_abs(args, device_id, cp, step)
 
    all_files = glob.glob(os.path.join('/content/PreSumm/bert_data/cnndm', '*'))
    print('Files In Input Dir: ' + str(len(all_files)))
    for file in all_files:
         with open(file) as f:
             source=f.read().rstrip()
 
         data_builder.str_format_to_bert(  source, args, '../bert_data_test/cnndm.test.0.bert.pt') 
         args.bert_data_path= '../bert_data_test/cnndm'
         test_text_abs(args, device_id, cp, step, predictor)
\ No newline at end of file