# Liver-Disease-Diagnostic
Machine learning codebase for 'Multiclass machine learning diagnostic for liver diseases by transcriptomics of 
peripheral blood mononuclear cells or liver tissue' publication in JHEP Reports.

ABSTRACT 

Background & Aims: Liver disease carries significant healthcare burden and frequently requires a combination of blood tests, imaging, and invasive liver biopsy to diagnose. Distinguishing between inflammatory liver diseases, which may have similar clinical presentations, is particularly challenging. In this study, we implemented a machine learning pipeline for the identification of diagnostic gene expression biomarkers across several alcohol-associated and non-alcohol-associated liver diseases, using either liver tissue or blood-based samples. 

Methods: We collected peripheral blood mononuclear cells (PBMCs) and liver tissue samples from participants with alcohol-associated hepatitis (AH), alcohol-associated cirrhosis (AC), non-alcohol-associated fatty liver disease, chronic hepatitis C infection, and healthy controls. We performed RNA sequencing (RNA-seq) for 137 PBMC samples and 67 liver tissue samples. Using gene expression data, we implemented a machine learning feature selection and classification pipeline to identify diagnostic biomarkers which distinguish between the liver disease groups. The liver tissue results were validated using a public independent RNA-seq dataset. The biomarkers were computationally validated for biological relevance using pathway analysis tools. 

Results: Utilizing liver tissue RNA-seq data, we distinguished between AH, AC, and healthy conditions with overall accuracies of 90% in our dataset, and 82% in the independent dataset, with 33 genes. Distinguishing four liver conditions and healthy controls yielded 91% overall accuracy in our liver tissue dataset with 39 genes, and 75% overall accuracy in our PBMC dataset with 75 genes. 

Conclusions: Our machine learning pipeline was effective at identifying a small set of diagnostic gene biomarkers and classifying several liver diseases using RNA-seq data from liver tissue and PBMCs. The methodologies implemented and genes identified in this study may facilitate future efforts toward a liquid biopsy diagnostic for liver diseases.

