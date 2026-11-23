if [ $1 == "RM" ]; then
	rm -rf ./visuals ./stats 'report.[pdf|aux|log|out|toc]' ./report.bbl ./report.blg ./pre-training_analysis
fi
