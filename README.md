#### Information and Instructions about the code

- To run the code, Python 3.6 and PyTorch 0.2 are needed
- The training code is present in the file `mtl_learning.py` whereas the testing code is present in the file `mtl_testing.py`
- The code in `mtl_learning.py` is pretty much self documented
- For the training, the datasets should be present as pickle files in the `pickles/` directory. The embeddings file should be present in the `data/` directory.
- In case of reloading the models and continuing the training, the models should be placed in `reloads/` directory.
- When all the necessary files and data are present, simply run `python mtl_learning.py` for training and `python mtl_testing.py` for testing.
- The outputs of the training would be present in a file named `outputs.txt` and that of testing would be present in `test_output.txt`

#### Credits

The project is inspired from Pasunuru et al's (2017) work on ["Towards Improving Abstractive Summarization via Entailment Generation"](http://www.aclweb.org/anthology/W17-4504) and Sean Robertson's [tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) on seq2seq translation.
