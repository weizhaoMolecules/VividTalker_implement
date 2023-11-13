python test_vq_decoder_speaker.py --config /model/HeapHeap/Learn2listen/learning2listen-main/src/configs/vq/trevor_speaker_delta_v6.json --checkpoint /model/HeapHeap/Learn2listen/learning2listen-main/src/models/speaker_trevor_er2er_best.pth --speaker 'trevor' --save


python test_vq_decoder_detail.py --config /model/HeapHeap/Learn2listen/learning2listen-main/src/configs/vq/trevor_speaker_delta_v6.json --checkpoint /model/HeapHeap/Learn2listen/learning2listen-main/src/models/detail/new_detail_trevor_er2er_best.pth --speaker 'trevor' --save


python test_vq_decoder_detail.py --config  /model/HeapHeap/Learn2listen/learning2listen-main/src/configs/vq/trevor_detail.json --checkpoint /model/HeapHeap/Learn2listen/learning2listen-main/src/models/detail/new_detail_trevor_er2er_best.pth --speaker trevor --save True



python test_vq_decoder_exp_only.py --config /model/HeapHeap/Learn2listen/learning2listen-main/src/configs/vq/trevor_exp_only.json --checkpoint /model/HeapHeap/Learn2listen/learning2listen-main/src/models/exp_only/only_exp_trevor_er2er_best.pth --speaker 'trevor'


python test_vq_decoder_detail.py --config /model/HeapHeap/Learn2listen/learning2listen-main/src/configs/vq/trevor_detail.json --checkpoint /model/HeapHeap/Learn2listen/learning2listen-main/src/models/detail/speaker_trevor_er2er_best.pth --speaker 'trevor'