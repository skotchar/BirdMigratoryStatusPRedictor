import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.DefaultXYDataset;

public class BirdMigratoryStatusPredictor extends JFrame {

    private JTextField filePathField;
    private JButton browseButton;
    private JComboBox<String> feature1ComboBox;
    private JComboBox<String> feature2ComboBox;
    private JPanel classifierButtonsPanel;
    private JTextArea resultArea;
    private JPanel chartPanelContainer;

    private Instances dataset;
    private List<String> numericFeatures;

    private boolean isUpdatingFeature1 = false;
    private boolean isUpdatingFeature2 = false;

    public BirdMigratoryStatusPredictor() {
        setTitle("Bird Migratory Status Predictor");
        setSize(1300, 800); 
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        numericFeatures = new ArrayList<>();

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new GridLayout(3, 1));

        JPanel filePanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel fileLabel = new JLabel("Dataset File:");
        filePathField = new JTextField(50);
        filePathField.setEditable(false);
        browseButton = new JButton("Browse");
        filePanel.add(fileLabel);
        filePanel.add(filePathField);
        filePanel.add(browseButton);
        topPanel.add(filePanel);

        JPanel featurePanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel feature1Label = new JLabel("Feature 1:");
        feature1ComboBox = new JComboBox<>();
        JLabel feature2Label = new JLabel("Feature 2:");
        feature2ComboBox = new JComboBox<>();
        featurePanel.add(feature1Label);
        featurePanel.add(feature1ComboBox);
        featurePanel.add(Box.createHorizontalStrut(20)); 
        featurePanel.add(feature2Label);
        featurePanel.add(feature2ComboBox);
        topPanel.add(featurePanel);

        JPanel classifierPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel classifierLabel = new JLabel("Select Classifier:");
        classifierButtonsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        classifierButtonsPanel.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        classifierButtonsPanel.setPreferredSize(new Dimension(1200, 50));

        String[] classifiers = {"Random Forest", "J48 (Decision Tree)", "Logistic Regression",
                "SMO (SVM)", "Naive Bayes", "IBk (k-NN)"};

        for (String clfName : classifiers) {
            JButton clfButton = new JButton(clfName);
            clfButton.addActionListener(new ClassifierButtonListener(clfName));
            classifierButtonsPanel.add(clfButton);
        }

        classifierPanel.add(classifierLabel);
        classifierPanel.add(classifierButtonsPanel);
        topPanel.add(classifierPanel);

        add(topPanel, BorderLayout.NORTH);

        JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);

        resultArea = new JTextArea();
        resultArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(resultArea);
        splitPane.setLeftComponent(scrollPane);

        chartPanelContainer = new JPanel(new BorderLayout());
        splitPane.setRightComponent(chartPanelContainer);
        splitPane.setDividerLocation(600);
        add(splitPane, BorderLayout.CENTER);

        browseButton.addActionListener(new BrowseButtonListener());
        feature1ComboBox.addItemListener(new FeatureSelectionListener(feature1ComboBox, feature2ComboBox, true));
        feature2ComboBox.addItemListener(new FeatureSelectionListener(feature2ComboBox, feature1ComboBox, false));
    }

    private class BrowseButtonListener implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            JFileChooser fileChooser = new JFileChooser();
            FileNameExtensionFilter filter = new FileNameExtensionFilter(
                    "CSV Files", "csv");
            fileChooser.setFileFilter(filter);
            int returnVal = fileChooser.showOpenDialog(BirdMigratoryStatusPredictor.this);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fileChooser.getSelectedFile();
                filePathField.setText(file.getAbsolutePath());
                try {

                    CSVLoader loader = new CSVLoader();
                    loader.setFieldSeparator(","); 
                    loader.setSource(file);
                    dataset = loader.getDataSet();

                    System.out.println("Loaded dataset: " + dataset.numInstances() + " instances, " + dataset.numAttributes() + " attributes.");

                    Attribute migratoryAttr = dataset.attribute("Migratory status");
                    if (migratoryAttr == null) {
                        throw new Exception("Attribute 'Migratory status' not found in the dataset.");
                    }
                    if (!migratoryAttr.isNominal()) {
                        throw new Exception("'Migratory status' attribute must be nominal.");
                    }
                    dataset.setClassIndex(migratoryAttr.index());

                    numericFeatures.clear();
                    feature1ComboBox.removeAllItems();
                    feature2ComboBox.removeAllItems();
                    for (int i = 0; i < dataset.numAttributes(); i++) {
                        if (i != dataset.classIndex() && dataset.attribute(i).isNumeric()) {
                            String attrName = dataset.attribute(i).name();
                            numericFeatures.add(attrName);
                            feature1ComboBox.addItem(attrName);
                            feature2ComboBox.addItem(attrName);
                        }
                    }

                    if (numericFeatures.isEmpty()) {
                        JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                                "No numeric features available in the dataset.",
                                "Error", JOptionPane.ERROR_MESSAGE);
                        return;
                    }

                    if (feature1ComboBox.getItemCount() > 0) {
                        feature1ComboBox.setSelectedIndex(0);
                    }
                    if (feature2ComboBox.getItemCount() > 1) {
                        feature2ComboBox.setSelectedIndex(1);
                    }

                } catch (Exception ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                            "Error loading dataset: " + ex.getMessage(),
                            "Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        }
    }

    private class FeatureSelectionListener implements ItemListener {
        private JComboBox<String> sourceComboBox;
        private JComboBox<String> targetComboBox;
        private boolean isFeature1;

        public FeatureSelectionListener(JComboBox<String> source, JComboBox<String> target, boolean isFeature1) {
            this.sourceComboBox = source;
            this.targetComboBox = target;
            this.isFeature1 = isFeature1;
        }

        @Override
        public void itemStateChanged(ItemEvent e) {
            if (e.getStateChange() == ItemEvent.SELECTED) {
                String selectedFeature = (String) sourceComboBox.getSelectedItem();
                if (selectedFeature == null) return;

                if (isFeature1 && isUpdatingFeature1) return;
                if (!isFeature1 && isUpdatingFeature2) return;

                try {
                    if (isFeature1) {
                        isUpdatingFeature2 = true;
                        updateTargetComboBox(targetComboBox, selectedFeature, feature1ComboBox.getSelectedItem());
                        isUpdatingFeature2 = false;
                    } else {
                        isUpdatingFeature1 = true;
                        updateTargetComboBox(targetComboBox, selectedFeature, feature2ComboBox.getSelectedItem());
                        isUpdatingFeature1 = false;
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }

        private void updateTargetComboBox(JComboBox<String> targetComboBox, String selectedFeature, Object currentSelection) {

            targetComboBox.removeAllItems();
            for (String feature : numericFeatures) {
                if (!feature.equals(selectedFeature)) {
                    targetComboBox.addItem(feature);
                }
            }

            if (currentSelection != null && !selectedFeature.equals(currentSelection.toString())) {
                targetComboBox.setSelectedItem(currentSelection.toString());
            } else if (targetComboBox.getItemCount() > 0) {
                targetComboBox.setSelectedIndex(0);
            }
        }
    }

    private class ClassifierButtonListener implements ActionListener {
        private String classifierName;

        public ClassifierButtonListener(String name) {
            this.classifierName = name;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            if (dataset == null) {
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "Please load a dataset first.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            String feature1 = (String) feature1ComboBox.getSelectedItem();
            String feature2 = (String) feature2ComboBox.getSelectedItem();

            if (feature1 == null) {
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "Please select at least one feature.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            if (feature2 != null && feature1.equals(feature2)) {
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "Please select two distinct features.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            try {

                ArrayList<Attribute> attributes = new ArrayList<>();
                Attribute attr1 = dataset.attribute(feature1);
                attributes.add(attr1);

                Attribute attr2 = null;
                if (feature2 != null) {
                    attr2 = dataset.attribute(feature2);
                    attributes.add(attr2);
                }

                Attribute classAttr = dataset.classAttribute();
                attributes.add(classAttr);

                Instances newData = new Instances("SelectedFeatures", attributes, dataset.numInstances());
                newData.setClassIndex(newData.numAttributes() - 1);

                for (int i = 0; i < dataset.numInstances(); i++) {
                    DenseInstance instance = new DenseInstance(newData.numAttributes());
                    instance.setDataset(newData);
                    instance.setValue(0, dataset.instance(i).value(attr1));

                    if (attr2 != null) {
                        instance.setValue(1, dataset.instance(i).value(attr2));
                        instance.setClassValue(dataset.instance(i).classValue());
                    } else {
                        instance.setClassValue(dataset.instance(i).classValue());
                    }
                    newData.add(instance);
                }

                int countResident = 0;
                int countMigratory = 0;
                for (int i = 0; i < newData.numInstances(); i++) {
                    double classValue = newData.instance(i).classValue();
                    String classLabel = newData.classAttribute().value((int) classValue);
                    if ("Resident".equalsIgnoreCase(classLabel)) {
                        countResident++;
                    } else if ("Migratory".equalsIgnoreCase(classLabel)) {
                        countMigratory++;
                    }
                }
                System.out.println("Before Resampling:");
                System.out.println("Resident: " + countResident);
                System.out.println("Migratory: " + countMigratory);

                Resample resample = new Resample();
                resample.setNoReplacement(false); 
                resample.setBiasToUniformClass(1.0); 
                resample.setInputFormat(newData);
                Instances balancedData = Filter.useFilter(newData, resample);

                int balancedResident = 0;
                int balancedMigratory = 0;
                for (int i = 0; i < balancedData.numInstances(); i++) {
                    double classValue = balancedData.instance(i).classValue();
                    String classLabel = balancedData.classAttribute().value((int) classValue);
                    if ("Resident".equalsIgnoreCase(classLabel)) {
                        balancedResident++;
                    } else if ("Migratory".equalsIgnoreCase(classLabel)) {
                        balancedMigratory++;
                    }
                }
                System.out.println("After Resampling:");
                System.out.println("Resident: " + balancedResident);
                System.out.println("Migratory: " + balancedMigratory);

                if (balancedResident == 0 || balancedMigratory == 0) {
                    JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                            "Resampling resulted in only one class. Please check your dataset.",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                Classifier classifier = getClassifierByName(classifierName);

                if (classifier == null) {
                    JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                            "Unsupported classifier selected.",
                            "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                Evaluation eval = new Evaluation(balancedData);
                eval.crossValidateModel(classifier, balancedData, 10, new Random(1));

                classifier.buildClassifier(balancedData);

                StringBuilder sb = new StringBuilder();
                sb.append("=== Evaluation Results ===\n");
                sb.append("Classifier: ").append(classifierName).append("\n\n");
                sb.append(eval.toSummaryString("\nResults\n======\n", false));
                sb.append(eval.toClassDetailsString());
                sb.append(eval.toMatrixString());

                resultArea.setText(sb.toString());

                generateROCCurve(classifier, balancedData, eval);

            } catch (Exception ex) {
                ex.printStackTrace();
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "Error during training: " + ex.getMessage(),
                        "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private Classifier getClassifierByName(String name) throws Exception {
        switch (name) {
            case "Random Forest":
                return new RandomForest();
            case "J48 (Decision Tree)":
                return new J48();
            case "Logistic Regression":
                return new Logistic();
            case "SMO (SVM)":
                return new SMO();
            case "Naive Bayes":
                return new NaiveBayes();
            case "IBk (k-NN)":
                return new IBk();
            default:
                return null;
        }
    }

    private void generateROCCurve(Classifier classifier, Instances data, Evaluation eval) {
        try {

            if (!data.classAttribute().isNominal() || data.classAttribute().numValues() != 2) {
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "ROC Curve can only be generated for binary classification.",
                        "Information", JOptionPane.INFORMATION_MESSAGE);
                return;
            }

            String positiveClassLabel = "Migratory";
            String negativeClassLabel = "Resident";

            int posIndex = data.classAttribute().indexOfValue(positiveClassLabel);
            int negIndex = data.classAttribute().indexOfValue(negativeClassLabel);

            if (posIndex == -1 || negIndex == -1) {
                JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                        "Class labels 'Migratory' and/or 'Resident' not found.",
                        "Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            List<Prediction> predictions = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                double[] dist = classifier.distributionForInstance(data.instance(i));
                double prob = dist[posIndex];
                double actual = data.instance(i).classValue() == posIndex ? 1.0 : 0.0;
                predictions.add(new Prediction(actual, prob));
            }

            predictions.sort((p1, p2) -> Double.compare(p2.prob, p1.prob));

            int totalPositives = (int) predictions.stream().filter(p -> p.actual == 1.0).count();
            int totalNegatives = predictions.size() - totalPositives;

            int truePositives = 0;
            int falsePositives = 0;

            List<double[]> rocPoints = new ArrayList<>();
            rocPoints.add(new double[]{0.0, 0.0}); 

            for (Prediction p : predictions) {
                if (p.actual == 1.0) {
                    truePositives++;
                } else {
                    falsePositives++;
                }

                double tpr = (double) truePositives / totalPositives; 
                double fpr = (double) falsePositives / totalNegatives; 

                rocPoints.add(new double[]{fpr, tpr});
            }

            rocPoints.add(new double[]{1.0, 1.0}); 

            DefaultXYDataset rocDataset = new DefaultXYDataset();
            double[][] dataArray = new double[2][rocPoints.size()];
            for (int i = 0; i < rocPoints.size(); i++) {
                dataArray[0][i] = rocPoints.get(i)[0]; 
                dataArray[1][i] = rocPoints.get(i)[1]; 
            }
            rocDataset.addSeries("ROC Curve", dataArray);

            JFreeChart chart = ChartFactory.createXYLineChart(
                    "ROC Curve",
                    "False Positive Rate",
                    "True Positive Rate",
                    rocDataset,
                    PlotOrientation.VERTICAL,
                    true,
                    true,
                    false
            );

            chart.getXYPlot().getRenderer().setSeriesPaint(0, Color.RED);
            chart.getXYPlot().getRenderer().setSeriesStroke(0, new BasicStroke(2.0f));

            chartPanelContainer.removeAll();
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanelContainer.add(chartPanel, BorderLayout.CENTER);
            chartPanelContainer.validate();

        } catch (Exception ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(BirdMigratoryStatusPredictor.this,
                    "Error generating ROC curve: " + ex.getMessage(),
                    "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private class Prediction {
        double actual;
        double prob;

        public Prediction(double actual, double prob) {
            this.actual = actual;
            this.prob = prob;
        }
    }

    public static void main(String[] args) {

        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
        }

        SwingUtilities.invokeLater(() -> {
            BirdMigratoryStatusPredictor app = new BirdMigratoryStatusPredictor();
            app.setVisible(true);
        });
    }
}