SUP_DIR="sup"
if [ ! -d "$SUP_DIR" ]; then
  echo "Creating $SUP_DIR"
  mkdir "$SUP_DIR"
fi

echo "Generating Supervised Data"

echo "Converting Train"
python convert_snips_to_assistant_format.py train.csv sup/train.tsv sup
echo "Converting Valid"
python convert_snips_to_assistant_format.py valid.csv sup/eval.tsv
echo "Converting Test"
python convert_snips_to_assistant_format.py test.csv sup/test.tsv

echo "Done Generating Supervised Data"

SPLIT_SEED=4242
NUMBER_OF_SPLITS=5
echo "Now Generating Unsupervised Data"
for frac in 0.25 0.40 0.75
do
  python convert_to_unsupervised_format.py ${frac} ${SPLIT_SEED} ${NUMBER_OF_SPLITS}
done
echo "Done Generating Unsupervised Data"
