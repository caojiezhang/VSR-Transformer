source /cluster/project/cvl/jiezcao/videosr/bin/activate

module load pigz
rsync -aq ./ ${TMPDIR}
tar -I pigz -xf /cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet.tar.gz -C ${TMPDIR}
tar -I pigz -xf /cluster/work/cvl/videosr/vimeo90k/vimeo_septuplet_matlabLRx4.tar.gz -C ${TMPDIR}
cd $TMPDIR

# Train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_vsrTransformer_x4_REDS.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/train_vsrTransformer_x4_Vimeo.yml --launcher pytorch

# Test
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_REDS.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_Vimeo.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/test_vsrTransformer_x4_Vid4.yml --launcher pytorch
