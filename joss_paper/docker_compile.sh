# sudo docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/paperdraft
sudo docker run -it --rm -v $(pwd):/data -u $(id -u):$(id -g) openjournals/inara -o pdf,crossref paper.md
