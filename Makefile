# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* servier/*.py

black:
	@black scripts/* servier/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr servier-*.dist-info
	@rm -fr servier.egg-info

install:
	@pip install .

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#     	  API & Streamlit
# ----------------------------------
streamlit_local:
	@streamlit run deploy/streamlit_app.py
api_local:
	@python deploy/api.py
# ----------------------------------
#     	DOCKER CMD
# ----------------------------------
clean_images:
	-@docker stop $$(docker ps -a -q)
	-@docker rm $$(docker ps -a -q)
	-@docker volume rm $$(docker volume ls -qf dangling=true)
	-@docker image rm $$(docker image ls -qf dangling=true)

docker_build_api:
	@docker build -f docker/Dockerfile.api -t servier-api .

docker_run_api:
	@docker run -d -p 8080:8080 servier-api

docker_build_streamlit:
	@docker build -f docker/Dockerfile.streamlit -t servier-streamlit .

docker_run_streamlit:
	@docker run -d -p 8501:8501 servier-streamlit

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2