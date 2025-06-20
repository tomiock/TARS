# Deep Learning Embeddings for Optimal Assignment of Translation Tasks


### Code execution
Before executing code, disable `wandb` by typing `wandb disable` on the terminal.

All of the files need to be executed from inside their respective folders. The main file for the training loop resides on `model`, from inside there execute `python3 embedding_model.py`.

Another important file would be `model/create_dataset.py` which created the dataframe used by the pytorch model.

### Plots
Some of the plots found in the report are created at `notebook/data_analysis.ipynb`. Most of them are from W&B.

### Other considerations
**IMPORTANT** if the github commits are taken into consideration:
> Towards the end of the project, the model was trained on the VM provided by the faculty. So the final commits are not reliable if taken into consideration, as the commits from the VM were made with the same account `tomiock`, regardless of who actually did the code.
