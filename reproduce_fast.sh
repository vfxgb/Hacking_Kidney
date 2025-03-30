for i in {0..4};do
  python train.py --use_sam --fold $i --nfolds 5 --batch_size 16 --epochs 50 --lr 1e-4 --expansion 0 --workers 16
done
cd ../
