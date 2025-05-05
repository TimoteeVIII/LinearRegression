# How to Use
- Load the Boston housing dataset into a database under a table called `housing` and db called `boston_db`
- Create another table in the same db called model_params with columns "weights", "bias", "learning_rate", "num_iteration", "regularisation_param", "convergence_limit", "training_mean", "training_std"
- The above is using Postgres
- Run all cells in `linearReg.ipynb`
- Start the docker engine
- Run `docker-compose up --build`
- Make a POST request to `http://127.0.0.1:8000/predict/` with a payload containing the following keys (all as numbers): crim, zn, indus, nox, rm, age, dis, rad, tax, pratio, b, lstat, chas
- A prediction should be returned

