"""Generates paper/beauty_classifier_audit.pdf: a CVPR-style writeup of the
scut_fbp_beauty_classifier.py / scut_fbp_beauty_cross_race_transfer.py investigation.

First-draft / just-for-fun document, not a peer-reviewed submission. All numbers below are
pulled directly from results/*.json produced by the scripts in this repo -- nothing here is
fabricated or estimated.
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, FrameBreak, Image, NextPageTemplate, PageBreak,
                                 PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.frames import Frame

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, '..', 'results')  # reuse the repo's tracked results/*.png (paper/figures/ is gitignored)
OUT_PDF = os.path.join(HERE, 'beauty_classifier_audit.pdf')

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
GUTTER = 0.28 * inch
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
CONTENT_H = PAGE_H - 2 * MARGIN

# ---------- styles ----------
styles = getSampleStyleSheet()

title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontName='Times-Bold',
                              fontSize=20, leading=24, spaceAfter=6, alignment=TA_CENTER)
author_style = ParagraphStyle('AuthorStyle', parent=styles['Normal'], fontName='Times-Roman',
                               fontSize=12, leading=15, alignment=TA_CENTER, spaceAfter=2)
affil_style = ParagraphStyle('AffilStyle', parent=styles['Normal'], fontName='Times-Italic',
                              fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=10,
                              textColor=colors.HexColor('#333333'))
abstract_head = ParagraphStyle('AbstractHead', parent=styles['Normal'], fontName='Times-Bold',
                                fontSize=11, alignment=TA_CENTER, spaceAfter=4, spaceBefore=8)
abstract_body = ParagraphStyle('AbstractBody', parent=styles['Normal'], fontName='Times-Roman',
                                fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY,
                                leftIndent=28, rightIndent=28)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Times-Bold', fontSize=12,
                     leading=14, spaceBefore=10, spaceAfter=4)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Times-Bold', fontSize=10.5,
                     leading=13, spaceBefore=8, spaceAfter=3)
body = ParagraphStyle('Body', parent=styles['Normal'], fontName='Times-Roman', fontSize=9,
                       leading=11.5, alignment=TA_JUSTIFY, spaceAfter=6)
caption = ParagraphStyle('Caption', parent=styles['Normal'], fontName='Times-Roman', fontSize=8.5,
                          leading=10.5, alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=12,
                          leftIndent=40, rightIndent=40)
footnote = ParagraphStyle('Footnote', parent=styles['Normal'], fontName='Times-Roman',
                           fontSize=7.5, leading=9.5, alignment=TA_JUSTIFY, textColor=colors.HexColor('#444444'))
ref_style = ParagraphStyle('Ref', parent=styles['Normal'], fontName='Times-Roman', fontSize=8.3,
                            leading=10.3, alignment=TA_JUSTIFY, spaceAfter=4,
                            leftIndent=10, firstLineIndent=-10)


def page_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Italic', 7.5)
    canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawCentredString(PAGE_W / 2, 0.45 * inch,
                              f'First draft -- not peer reviewed -- {canvas.getPageNumber()}')
    canvas.restoreState()


def fit_image(path, max_w, max_h):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = min(max_w / w, max_h / h)
    return Image(path, width=w * scale, height=h * scale)


def make_doc():
    doc = BaseDocTemplate(OUT_PDF, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
                           topMargin=MARGIN, bottomMargin=MARGIN, title='What Does a "Beauty" Classifier Actually Learn?')
    full_frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, CONTENT_H, id='full')
    col1 = Frame(MARGIN, MARGIN, COL_W, CONTENT_H, id='col1')
    col2 = Frame(MARGIN + COL_W + GUTTER, MARGIN, COL_W, CONTENT_H, id='col2')
    doc.addPageTemplates([
        PageTemplate(id='Full', frames=[full_frame], onPage=page_header_footer),
        PageTemplate(id='TwoCol', frames=[col1, col2], onPage=page_header_footer),
    ])
    return doc


def P(text, style=body):
    return Paragraph(text, style)


def section(title):
    return P(title, h1)


def subsection(title):
    return P(title, h2)


story = []

# ---------------- Title page ----------------
story.append(P('What Does a &ldquo;Beauty&rdquo; Classifier Actually Learn?<br/>'
               'An Empirical Audit of Cross-Population Generalization on SCUT-FBP5500', title_style))
story.append(Spacer(1, 6))
story.append(P('Claude Anthropsky<sup>1</sup> &nbsp;&nbsp; Boris Epshtein<sup>2,&#8224;</sup>', author_style))
story.append(P('<sup>1</sup>AI Research Assistant (Claude, Anthropic) &nbsp;&nbsp; '
               '<sup>2</sup>Independent Researcher<br/>'
               '<sup>&#8224;</sup>Corresponding author: boris.epshtein@gmail.com', affil_style))
story.append(P(
    'This paper was drafted collaboratively by a human and an AI assistant (Claude Sonnet 5, by '
    'Anthropic) as a same-day writeup of an exploratory investigation conducted together in a '
    'personal GitHub repository. It is a first draft, has not been peer reviewed, and is not an '
    'official publication of Anthropic. "Claude Anthropsky" is a playful byline, not a claim of '
    'institutional affiliation or authorship in the legal sense.', footnote))
story.append(Spacer(1, 8))

story.append(P('Abstract', abstract_head))
story.append(P(
    'Motivated by the folk claim that &ldquo;beauty is in the eye of the beholder,&rdquo; we train a '
    'ResNet18 classifier to distinguish faces rated &ldquo;pretty&rdquo; from faces rated &ldquo;average&rdquo; '
    'on the Caucasian-female (CF) subset of SCUT-FBP5500, using a median split on the dataset&rsquo;s '
    'mean human beauty ratings. A single 70/15/15 split reaches 89.4% test accuracy and 0.960 ROC AUC '
    '&mdash; a result that felt suspiciously easy and prompted a series of falsification attempts rather '
    'than a declaration of victory. We show the result is stable under 5-fold cross-validation '
    '(accuracy 0.875&plusmn;0.026, AUC 0.944&plusmn;0.012), then test whether it reflects a population-general '
    'signal by training on one race/gender subset and evaluating on another. We find a striking '
    'asymmetry: a model trained on Asian-female (AF) faces transfers to CF faces with almost no loss '
    '(in-domain AUC 0.927 vs. cross-domain AUC 0.914), while a model trained on CF faces transfers to '
    'AF faces with severely degraded calibration (cross-domain accuracy collapses to 0.505, chance '
    'level) even though its rank-ordering partially survives (cross-domain AUC 0.832). A controlled '
    're-run with AF capped at CF&rsquo;s training-set size rules out sample size as the explanation. We reject '
    'an oval-crop framing-artifact hypothesis outright, and find mixed evidence on hair: a coarse '
    'regional hue/saturation/brightness proxy shows no strong effect, but a direct causal test &mdash; '
    'cropping every face to its facial-landmark bounding box to exclude hair almost entirely, after a '
    'flawed grayscale-ablation proposal was caught and corrected mid-project &mdash; raises cross-domain '
    'accuracy from 0.505 (chance) to 0.596 and AUC from 0.832 to 0.867, showing hair region was part of '
    'the shortcut without fully explaining it. Age, estimated with an off-the-shelf model, turned out to '
    'be the strongest single correlate found (r=&minus;0.494 with the pretty label, a 6.6-year mean gap) '
    '&mdash; yet retraining on an age-matched subset barely changed AUC (0.958 vs. 0.958 unmatched), so '
    'this strong data-level confound does not appear to be what the model is actually using. Finally, '
    'adding two datasets with genuinely different rater pools (Chicago Face Database, Face Research '
    'Lab London Set) and running leave-one-source-out cross-validation, every held-out direction beats '
    'chance (one with identical 0.750 in-/out-of-domain AUC), but calibration collapse recurs '
    'elsewhere (accuracy as low as 0.559) and remains unsolved. We do not claim to resolve whether '
    'beauty is objective; we show that a plausible-looking, statistically stable classifier can still '
    'be poorly calibrated across populations for reasons only partly explained by any single mechanism '
    'we tested, and that most of our own hypotheses about why did not survive contact with the full '
    'dataset.', abstract_body))

story.append(Spacer(1, 4))
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty_rating_scale_examples.png'), 3.6 * inch, 2.3 * inch))
story.append(P(
    '<b>Figure 1.</b> What the label actually means: four CF images closest to each integer mean rating '
    '(shown above each face), from the least-attractive to the most-attractive end of the scale. No '
    'image is rated exactly 1 or 5 &mdash; individual scores are means of 60 raters and rarely reach the '
    'scale&rsquo;s endpoints; these are the closest available. The median split used throughout this report '
    '(&sect;3) falls at 2.97, between the second and third rows.', caption))

story.append(NextPageTemplate('TwoCol'))
story.append(PageBreak())

# ---------------- 1. Introduction ----------------
story.append(section('1. Introduction'))
story.append(P(
    'Whether physical attractiveness reflects an objective, shared property of faces or a purely '
    'subjective, culturally contingent judgment is an old debate. A modern, empirical way to poke at '
    'it is to ask: can a model learn to predict human attractiveness ratings at all, and if so, does '
    'what it learns generalize across the population of raters and ratees, or is it specific to the '
    'population it was trained on? The first question is answerable with a standard supervised-learning '
    'setup; the second requires deliberately trying to break the first answer.', body))
story.append(P(
    'This report documents exactly that process on the SCUT-FBP5500 dataset '
    '[<a href="#ref1" color="blue">1</a>]: we build a baseline classifier, get a result that looks '
    'almost too good, and then spend most of the paper trying to explain it away. Several explanations '
    'we proposed turned out to be wrong once checked against the full data rather than a handful of '
    'examples; we report those failures alongside the results that held up, because in an informal '
    'audit like this one the failed hypotheses are as informative as the successful ones.', body))
story.append(P(
    'We want to be precise about what this is and is not. It is not a claim that beauty is or is not '
    'objective &mdash; a classifier reproducing the statistical regularities in 60 raters&rsquo; judgments shows '
    'that those judgments have learnable structure, not that the structure reflects a mind-independent '
    'property of faces. It is an audit of a specific, small, non-commercial-research-license dataset and '
    'a specific, small model, run over the course of one afternoon. We report it in this format for fun, '
    'and because the negative results (see &sect;4.3, &sect;4.6) are a reasonably good illustration of how easy '
    'it is to convince yourself of a spurious explanation from a handful of cherry-picked examples.', body))

story.append(section('2. Related Work'))
story.append(P(
    'SCUT-FBP5500 [<a href="#ref1" color="blue">1</a>] was released specifically for facial beauty '
    'prediction research, with mean attractiveness ratings from 60 volunteers over 5,500 Asian and '
    'Caucasian, male and female faces. We use its bundled 86-point facial landmarks '
    '(&sect;4.7) and its official label file rather than an older, differently-named label file that '
    'ships in the dataset&rsquo;s GitHub repository but does not match this archive&rsquo;s image filenames '
    '&mdash; a mismatch that cost us one failed run before we noticed it. Our attention-visualization '
    'method is Grad-CAM [<a href="#ref2" color="blue">2</a>], applied to the last convolutional block of '
    'a ResNet18 [<a href="#ref3" color="blue">3</a>] pretrained on ImageNet.', body))
story.append(P(
    'Our central empirical finding &mdash; that a model trained on SCUT-FBP5500 generalizes poorly '
    'outside the population it was trained on &mdash; is not new. Fernandes et al. '
    '[<a href="#ref4" color="blue">4</a>] cite prior work by Zejmo et al. '
    '[<a href="#ref5" color="blue">5</a>] showing that beauty-prediction models trained on SCUT-FBP5500 '
    'predict poorly on other face databases, and separately note the dataset&rsquo;s all-Asian rater pool as '
    'a limitation for generalization, motivating their own multi-database training approach (see '
    '&sect;7 for why this rater-pool fact also changes how our &sect;4.4&ndash;4.7 results should be read). '
    'MEBeauty and similar more ethnically diverse beauty-rating datasets exist precisely to address this '
    'gap. We therefore make no novelty claim for the qualitative phenomenon (poor cross-population '
    'generalization); our contribution, such as it is, is a specific, transparently-reported audit of '
    '<i>why</i> on this dataset and this model, tracking several candidate mechanisms to a partial answer.',
    body))

story.append(section('3. Method'))
story.append(P(
    'Dataset. SCUT-FBP5500 filenames encode race and gender as a two-letter prefix; we use the '
    'Caucasian-female (CF, n=750) and Asian-female (AF, n=2000) subsets. Each image has a mean beauty '
    'rating on a 1&ndash;5 scale. We binarize within each subset independently via a median split '
    '(score above that subset&rsquo;s own 50th percentile &rarr; &ldquo;pretty&rdquo;), so cross-population comparisons '
    'are not skewed by a possible rating-scale offset between populations.', body))
story.append(P(
    'Model and training. A ResNet18 pretrained on ImageNet, with the final fully-connected layer '
    'replaced by a single logit and trained with binary cross-entropy (Adam, lr=1e-4). We use early '
    'stopping (minimum 8 / maximum 40 epochs, patience 5 epochs, 1% minimum relative improvement in '
    'validation loss) on a held-out validation split, and report the checkpoint with the best validation '
    'loss. Images are resized to 224&times;224 and normalized with standard ImageNet statistics; training '
    'uses random horizontal flip as the only augmentation. Sigmoid output is used directly as a '
    '&ldquo;beauty score.&rdquo; All code and raw per-run metrics are version-controlled alongside this paper.', body))

# ---------------- Figure 2 (full width) ----------------
story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty.png'), PAGE_W - 2 * MARGIN, 8.1 * inch))
story.append(P(
    '<b>Figure 2.</b> Baseline CF classifier, 5-fold cross-validation. Top: training/validation loss '
    '(fold 1). Rows 2&ndash;5: the five highest- and five lowest-scoring test faces (fold 1) with '
    'Grad-CAM overlays &mdash; attention on &ldquo;pretty&rdquo; predictions concentrates on central facial '
    'structure (nose, cheeks, mouth); attention on &ldquo;average&rdquo; predictions is weaker/more diffuse and '
    'sometimes lands on the hairline/forehead band rather than any single feature (&sect;4.3). Bottom: '
    'ROC curves for all 5 folds (mean AUC 0.944&plusmn;0.012) and per-fold accuracy/AUC bars.', caption))
story.append(NextPageTemplate('TwoCol'))
story.append(PageBreak())

# ---------------- 4. Experiments ----------------
story.append(section('4. Experiments and Results'))

story.append(subsection('4.1 Baseline'))
story.append(P(
    'A single stratified 70/15/15 split of the CF subset (525/112/113 images) reaches 89.4% test '
    'accuracy and 0.960 ROC AUC, with mean predicted score 0.815 for &ldquo;pretty&rdquo; faces vs. 0.107 for '
    '&ldquo;average&rdquo; faces &mdash; a clean separation that struck us as suspicious rather than reassuring for '
    'a 750-image dataset with a hand-picked binary threshold.', body))

story.append(subsection('4.2 Attention and a False Lead'))
story.append(P(
    'Grad-CAM on the five highest- and lowest-scoring test faces showed the expected pattern for '
    '&ldquo;pretty&rdquo; predictions (central facial structure) but a mix of weak/diffuse and off-face '
    '(image-corner) attention for &ldquo;average&rdquo; predictions. Eyeballing those ten images also suggested '
    'a framing confound: the &ldquo;pretty&rdquo; examples were full-frame photographs, the &ldquo;average&rdquo; '
    'examples were oval-vignette headshots. Checked against the full 750-image subset, this did not '
    'hold up: 96% of &ldquo;pretty&rdquo; images and 100% of &ldquo;average&rdquo; images have the same flat-corner '
    'vignette signature, and a random 24-image sample confirmed both framing styles occur in both '
    'classes. A follow-up check of sharpness (Laplacian variance, as a retouching proxy), saturation, '
    'and brightness in a center-face crop found only modest correlations with the label (r=0.20, '
    'r=&minus;0.12, r=0.09 respectively) &mdash; no single dominant artifact, more likely several weak, '
    'correlated cues combined nonlinearly by the network.', body))

story.append(subsection('4.3 Stability (5-fold CV)'))
story.append(P(
    'A single split&rsquo;s 0.960 AUC could be a fluke of a favorable 113-image test set. Five-fold '
    'stratified cross-validation (a fresh ResNet18 per fold) gives accuracy 0.875&plusmn;0.026 and ROC AUC '
    '0.944&plusmn;0.012 (fold range: 0.927&ndash;0.959) &mdash; see Figure 2. The result is stable, not a fluke of '
    'one split, though the original single-split AUC was near the best fold rather than the mean.', body))

story.append(subsection('4.4 Cross-Population Transfer'))
story.append(P(
    'The most direct test of whether the learned distinction generalizes beyond its training '
    'population: train on one race/gender subset, evaluate in-domain (held-out faces from the same '
    'subset) and cross-domain (the full other subset, labeled by its own median split). Table 1 '
    'summarizes all runs; Figure 3 shows ROC curves and Grad-CAM on the cross-domain extremes.', body))
story.append(P(
    'Trained on CF, in-domain AUC is 0.958, but cross-domain AUC on the full AF set (n=2000) drops to '
    '0.831&ndash;0.832 and accuracy collapses to 0.505 &mdash; chance level. The mean predicted score for both '
    'AF classes is near 1.0 (0.988 for true-pretty, 0.933&ndash;0.934 for true-average): the model does not '
    'lose its ability to rank AF faces relative to each other so much as it loses calibration entirely, '
    'defaulting to &ldquo;pretty&rdquo; almost everywhere. Trained on AF, the reverse direction transfers almost '
    'perfectly: in-domain AUC 0.927&ndash;0.928, cross-domain AUC on the full CF set (n=750) 0.914&ndash;0.9140, '
    'with mean scores staying well separated (0.84&ndash;0.87 vs. 0.25&ndash;0.31).', body))

story.append(subsection('4.5 Ruling Out Sample Size'))
story.append(P(
    'AF has 2.7&times; more images than CF (2000 vs. 750); before attributing the asymmetry to '
    'population differences, we re-ran the AF&rarr;CF direction with AF capped at 750 images (stratified '
    'subsample), giving an identical training-set size (525, after the same 70/15/15 split) to the '
    'CF-trained model. Result: in-domain AUC 0.983, cross-domain AUC 0.909, accuracy 0.815 &mdash; '
    'essentially unchanged from the full AF(2000)&rarr;CF run. Sample size does not explain the asymmetry.', body))

story.append(subsection('4.6 A Hair-Color Hypothesis, Tested and Not Supported'))
story.append(P(
    'One candidate explanation: CF faces have far more hair/eye color variation than AF faces (mostly '
    'dark hair), so a CF-trained model could lean on color as a shortcut that simply does not exist in '
    'AF. We measured HSV statistics in a coarse top-of-image &ldquo;hair region&rdquo; band and correlated them '
    'with the label. Within CF, brightness correlated negatively with &ldquo;pretty&rdquo; (r=&minus;0.165: pretty '
    'faces had a <i>darker</i> hair-region reading, opposite the naive &ldquo;blonde = pretty&rdquo; story); '
    'saturation and hue correlations were weaker still (r=0.068, r=0.094). Critically, the '
    'cross-population variance comparison contradicted the premise: AF showed <i>higher</i> saturation '
    'variance (44.8) than CF (31.2), not lower. This simple proxy did not support the hypothesis.', body))

story.append(subsection('4.7 Hair-Exclusion Ablation'))
story.append(P(
    'A regional color average can miss a spatially-localized cue a CNN could still exploit, so we '
    'designed a more direct test: crop every face to its 86-point facial-landmark bounding box '
    '(eyebrows to chin, cheek to cheek), which excludes hair almost entirely (Figure 4), and re-run the '
    'CF&harr;AF transfer test on the crops. An initial proposal to test this via grayscale conversion was '
    'incorrect and was caught before implementation: grayscale removes hue/chroma but preserves '
    'luminance, and hair lightness (the main axis of the color hypothesis) is primarily a luminance '
    'signal &mdash; blonde hair stays visibly light and dark hair stays visibly dark after desaturation, so '
    'a grayscale ablation would not have tested what it claimed to.', body))
story.append(P(
    'Result (Figure 5): excluding hair <b>meaningfully improved, but did not fix,</b> CF&rarr;AF '
    'calibration. Cross-domain accuracy rose from 0.505 (chance) to 0.596, cross-domain AUC rose from '
    '0.832 to 0.867, and &mdash; the clearest sign of a real effect &mdash; the mean score for true-average AF '
    'faces dropped from 0.933 (barely distinguishable from true-pretty&rsquo;s 0.988) to 0.748 (vs. '
    '0.970), a real widening of the separation. The already-good AF&rarr;CF direction was essentially '
    'unaffected (cross-domain AUC 0.914&rarr;0.921). So hair region was part of the CF&rarr;AF shortcut, '
    'but not the whole story: cross-domain accuracy (0.596) remains far below in-domain (0.894) even '
    'with hair excluded. This directly contradicts our own &sect;4.6 proxy check, which found no strong '
    'hair-color signal &mdash; a spatially-localized cue a coarse regional pixel average can miss entirely. '
    'Grad-CAM on the hair-excluded crops also shows a qualitative shift: &ldquo;average&rdquo;-predicted AF '
    'faces now draw attention to the mouth/jaw region rather than the hairline or image corners seen '
    'in earlier figures &mdash; a new, as yet unexplained, observation.', body))

# ---------------- Figure 3 ----------------
story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty_cross_race_transfer.png'),
                        5.6 * inch, 8.6 * inch))
story.append(P(
    '<b>Figure 3.</b> Cross-population transfer. Top: trained on CF, in-domain (blue) vs. cross-domain '
    'AF (red) ROC, with Grad-CAM on AF examples the model scores highest/lowest. Middle: trained on the '
    'full AF set, in-domain vs. cross-domain CF ROC. Bottom: trained on AF capped to CF&rsquo;s training-set '
    'size (n=750), same comparison &mdash; nearly identical to the full-AF run, ruling out sample size as '
    'the explanation for the CF&rarr;AF asymmetry (&sect;4.5).', caption))
# ---------------- Table 1 ----------------
story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(P('<b>Table 1.</b> Summary of all reported runs. Accuracy/AUC on the classifier&rsquo;s own '
               'test set (&ldquo;in-domain&rdquo;) vs. the full other-population subset (&ldquo;cross-domain&rdquo;), '
               'each labeled by that population&rsquo;s own median split.', caption))
table_data = [
    ['Run', 'Train (n)', 'In-domain\nAcc / AUC', 'Cross-domain\nAcc / AUC', 'Notes'],
    ['CF baseline (single split)', '525', '0.894 / 0.960', 'n/a', 'felt too easy (§4.1)'],
    ['CF baseline (5-fold CV mean)', '~510', '0.875±.026 / 0.944±.012', 'n/a', 'stable across folds (§4.3)'],
    ['Trained CF, tested AF', '525', '0.885 / 0.958', '0.505 / 0.832', 'calibration collapse (§4.4)'],
    ['Trained AF(2000), tested CF', '1400', '0.833 / 0.927', '0.824 / 0.914', 'transfers cleanly (§4.4)'],
    ['Trained AF(750), tested CF', '525', '0.920 / 0.983', '0.815 / 0.909', 'rules out data volume (§4.5)'],
    ['Trained CF, tested AF (no hair)', '525', '0.894 / 0.971', '0.596 / 0.867', 'hair partly explains it (§4.7)'],
    ['Trained AF, tested CF (no hair)', '1400', '0.867 / 0.936', '0.820 / 0.921', 'unaffected by crop (§4.7)'],
    ['CF random subsample (n=452)', '316', '0.926 / 0.955', 'n/a', 'sample-size control (§4.8)'],
    ['CF age-matched subsample (n=452)', '316', '0.868 / 0.958', 'n/a', 'age barely matters (§4.8)'],
]
tbl = Table(table_data, colWidths=[1.7 * inch, 0.7 * inch, 1.35 * inch, 1.35 * inch, 1.6 * inch])
tbl.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
    ('LEADING', (0, 0), (-1, -1), 9.5),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
    ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.black),
    ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(tbl)
story.append(Spacer(1, 10))
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty_landmark_crop_illustration.png'),
                        PAGE_W - 2 * MARGIN, 2.6 * inch))
story.append(P(
    '<b>Figure 4.</b> Method for &sect;4.7: original images (top) and their 86-point-landmark face crop '
    '(bottom), which excludes hair almost entirely except for some residual fringe/bangs leakage in a '
    'minority of hairstyles where hair hangs low over the forehead, inside the landmark bounding box.',
    caption))

story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty_cross_race_transfer_no_hair.png'),
                        6.6 * inch, 8.4 * inch))
story.append(P(
    '<b>Figure 5.</b> Cross-population transfer with hair excluded (&sect;4.7), same layout as Figure 3. '
    'Top: trained on CF, cross-domain AF AUC rises to 0.867 (from 0.832 with hair) and the '
    '&ldquo;average&rdquo;-predicted AF examples now draw Grad-CAM attention to the mouth/jaw rather than the '
    'hairline or image corners seen with hair included. Bottom: the already-good AF&rarr;CF direction is '
    'essentially unchanged.', caption))
story.append(NextPageTemplate('TwoCol'))
story.append(PageBreak())

story.append(subsection('4.8 Age: The Strongest Confound Found, With a Twist'))
story.append(P(
    'Age was flagged twice earlier (the hairline/forehead Grad-CAM signal in &sect;4.3; visibly older '
    'faces among random &ldquo;average&rdquo; examples in early qualitative checks) but never measured. We '
    'estimated age for every CF and AF image with an off-the-shelf age model, DeepFace '
    '[<a href="#ref6" color="blue">6</a>], then built an age-matched subset (equal pretty/average counts '
    'within each 5-year age bin, so the two label groups share an age distribution by construction) and '
    'retrained the baseline recipe on it, alongside a same-size random (not age-matched) control.', body))
story.append(P(
    'Age turned out to be by far the strongest single correlate we measured all day: r=&minus;0.494 '
    'between age and the &ldquo;pretty&rdquo; label (CF &ldquo;pretty&rdquo; faces average 29.6 years vs. '
    '&ldquo;average&rdquo; faces at 36.2 &mdash; a 6.6-year gap), dwarfing sharpness (r=0.20), saturation '
    '(r=&minus;0.12), and brightness (r=0.09) from &sect;4.2. AF is also both younger (mean 29.0y) and far '
    'more age-homogeneous (std 3.8y) than CF (mean 33.0y, std 6.7y) &mdash; a plausible additional piece of '
    'the CF&rarr;AF calibration story from &sect;4.4: a &ldquo;younger=pretty&rdquo; cue learned across CF&rsquo;s wide age '
    'range would have little room to discriminate within AF&rsquo;s narrow one. We did not test this '
    'directly with an age-matched cross-population retrain, so it remains a plausible reading rather '
    'than a demonstrated one.', body))
story.append(P(
    'The twist (Figure 6): despite that strong correlation, age-matched retraining barely moved the '
    'model. ROC AUC was 0.958 for the full baseline, 0.955 for the same-size random-subsample control, '
    'and 0.958 for the age-matched subset &mdash; essentially identical. Accuracy dropped modestly (0.894 '
    '&rarr; 0.868) but within the noise of a 68-image test set, and not clearly below the random '
    'control&rsquo;s 0.926. So age is a real, strong pattern in the underlying <i>human ratings</i> &mdash; '
    'younger CF faces genuinely were rated more attractive on average by this rater pool &mdash; but the '
    '<i>model</i> does not appear to be leaning on it much: it keeps most of its discriminative power even '
    'when age can&rsquo;t be used as a shortcut. This is the clearest instance in this report of a '
    'candidate confound that is unambiguously real in the data without turning out to be what the '
    'classifier is actually doing.', body))

story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(fit_image(os.path.join(FIG_DIR, 'scut_fbp_beauty_age_control.png'), 7.0 * inch, 6.6 * inch))
story.append(P(
    '<b>Figure 6.</b> Age-controlled ablation (&sect;4.8). Top: CF age distribution by label before '
    'matching (6.6-year gap between means). Middle: after age-matching (means within 0.4 years, '
    'n=452). Bottom: accuracy/AUC for the full baseline vs. a same-size random subsample vs. the '
    'age-matched subsample &mdash; AUC is essentially unchanged across all three despite the large age '
    'gap being fully controlled for in the third condition.', caption))
story.append(NextPageTemplate('TwoCol'))
story.append(PageBreak())

# ---------------- 5. Toward a Better Classifier ----------------
story.append(section('5. Toward a Better Classifier: Cross-Rater-Culture Training'))
story.append(P(
    'Everything above is an audit of one classifier; at this point the corresponding author asked '
    'for something more useful than an audit &mdash; a classifier that does not pick up the wrong '
    'cues. Section&nbsp;7 notes that CF and AF share a single rater culture (SCUT&rsquo;s 60 Asian '
    'raters), so &sect;4.4&ndash;4.8 cannot speak to whether judgments transfer across genuinely '
    'independent rater pools. We added two more independently-rated datasets to test exactly that: '
    'the Chicago Face Database (CFD) [<a href="#ref7" color="blue">7</a>], 827 individuals rated '
    '1&ndash;7 by roughly 2,500 U.S. raters across three release waves (597 White/Black/Asian/Latino, '
    '88 multiracial, 142 India), and the Face Research Lab London Set '
    '[<a href="#ref8" color="blue">8</a>], 102 individuals (69 white, 13 black, 10 west-Asian, 9 '
    'east-Asian, 1 mixed) rated 1&ndash;7 by 2,513 raters aged 17&ndash;90. Restricted to female '
    'subjects for comparability with &sect;4: SCUT contributes 2,750 (CF+AF pooled), CFD 410, London '
    '49 &mdash; a size imbalance we return to below. Every image was cropped to face-only before '
    'training, given &sect;4.7&rsquo;s finding that hair region was part of the CF&rarr;AF shortcut: '
    'SCUT via its own 86-point landmarks as before, CFD/London via YuNet '
    '[<a href="#ref9" color="blue">9</a>], a lightweight CNN face detector bundled with OpenCV, '
    'chosen after two other detectors (a DeepFace backend, then MediaPipe) each failed with an '
    'unrelated library-compatibility crash on the Colab runtime used for training.', body))
story.append(P(
    'Validation is leave-one-source-out cross-validation: train on two sources, evaluate '
    'in-domain (a held-out slice of those two) and out-of-domain (the fully-held-out third source, '
    'in its entirety) for all three combinations, then train one final model on everything '
    'combined. Table&nbsp;2 and Figure&nbsp;7 summarize the results.', body))
story.append(P(
    'Every held-out direction beats chance by a real margin &mdash; genuine evidence of a '
    'transferable signal across three independent rater cultures, not just across race within one '
    'culture. The CFD+London&rarr;SCUT direction is the strongest evidence: AUC is 0.750 both '
    'in-domain and out-of-domain, meaning a model trained on only 321 images from the two smaller, '
    'more diverse sources ranks the full 2,750-image SCUT set exactly as well as it ranks its own '
    'held-out test data. But the same calibration problem from &sect;4.4 recurs: accuracy lags AUC '
    'out-of-domain in the other two directions, worst for SCUT+London&rarr;CFD (AUC drops from 0.946 '
    'in-domain to 0.676 out-of-domain, accuracy to 0.559 &mdash; barely above chance). CFD is the '
    'weak point throughout: it is both the hardest target to transfer to and, in the final combined '
    'model, the source with the lowest per-source test AUC (0.814 vs. SCUT&rsquo;s 0.944) despite '
    'being part of that model&rsquo;s own training data. We suspect this is related to the size '
    'imbalance noted above &mdash; SCUT is 86% of the combined pool by volume, so the combined model '
    'is closer to &ldquo;SCUT with extra data&rdquo; than a balanced three-culture model &mdash; but '
    'have not yet isolated the mechanism; that is the immediate next thing we are chasing.', body))

table2_data = [
    ['Held out', 'Train\n(n)', 'In-domain\nAcc/AUC', 'Out-of-domain\nAcc/AUC'],
    ['SCUT (train CFD+London)', '321', '0.667/0.750', '0.577/0.750'],
    ['CFD (train SCUT+London)', '1959', '0.871/0.946', '0.559/0.676'],
    ['London (train SCUT+CFD)', '2212', '0.852/0.940', '0.673/0.827'],
]
story.append(P('<b>Table 2.</b> Leave-one-source-out cross-validation. &ldquo;Out-of-domain&rdquo; '
               'evaluates on the entire held-out source, not a subsample.', caption))
tbl2 = Table(table2_data, colWidths=[1.35 * inch, 0.35 * inch, 0.75 * inch, 0.75 * inch])
tbl2.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
    ('FONTSIZE', (0, 0), (-1, -1), 7.3),
    ('LEADING', (0, 0), (-1, -1), 8.8),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
    ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.black),
    ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(tbl2)
story.append(P(
    'The final all-sources-combined model reaches 0.851 accuracy / 0.928 AUC overall (n=482 test), '
    'but the per-source breakdown shows the imbalance directly: 0.880/0.944 on its SCUT test slice '
    '(n=409) vs. 0.716/0.814 on its CFD slice (n=67); London&rsquo;s slice was too small (&lt;10) to '
    'report separately. The Grad-CAM panel in Figure&nbsp;7 is entirely SCUT examples for the same '
    'reason &mdash; SCUT&rsquo;s volume dominates the test set&rsquo;s extreme scores, so this run '
    'gives no visual read on what the model attends to in CFD or London faces.', body))

story.append(NextPageTemplate('Full'))
story.append(PageBreak())
story.append(fit_image(os.path.join(FIG_DIR, 'beauty_classifier_multisource.png'), 5.4 * inch, 8.4 * inch))
story.append(P(
    '<b>Figure 7.</b> Leave-one-source-out cross-validation (top three panels) and the final '
    'combined-source model (fourth panel + Grad-CAM strip). Every out-of-domain ROC beats chance; '
    'CFD+London&rarr;SCUT (top) shows in-domain and out-of-domain curves nearly overlapping. The '
    'Grad-CAM examples are all SCUT because SCUT dominates the combined test set by volume '
    '(&sect;5).', caption))
story.append(NextPageTemplate('TwoCol'))
story.append(PageBreak())

# ---------------- 6. Discussion ----------------
story.append(section('6. Discussion'))
story.append(P(
    'What does this say about whether beauty is objective? Less than either a strict subjectivist or '
    'a strict objectivist would like. The AF&rarr;CF direction transferring almost perfectly argues '
    'against &ldquo;these judgments are purely population-specific and share nothing across groups.&rdquo; The '
    'CF&rarr;AF direction losing calibration (while keeping a well-above-chance AUC) argues against '
    '&ldquo;the learned signal is a population-independent, objective beauty detector&rdquo; in any strong sense '
    '&mdash; if it were, both directions should transfer symmetrically. The asymmetry itself is the most '
    'interesting finding in this report. Of the mechanisms we tested, framing and sample size explain '
    'none of it; hair region (&sect;4.7) explains some of it (cross-domain accuracy rose 9 points once '
    'hair was excluded) but far from all of it (0.596 cross-domain vs. 0.894 in-domain remains a large '
    'gap). Age (&sect;4.8) is a plausible contributor to the asymmetry specifically &mdash; AF is both '
    'younger and far more age-homogeneous than CF, so a &ldquo;younger=pretty&rdquo; cue learned on CF would have '
    'little room to discriminate within AF &mdash; but we did not test this directly, and it sits '
    'awkwardly next to age-matching barely denting the CF-only baseline&rsquo;s AUC. The residual gap, and '
    'the new mouth/jaw Grad-CAM signal that appeared once hair was removed, remain unexplained and would '
    'be the natural next thing to chase.', body))
story.append(P(
    'A methodological point we think is worth stating plainly: two of our own hypotheses (the '
    'oval-crop framing artifact in &sect;4.2, and the coarse-proxy version of the hair-color hypothesis in '
    '&sect;4.6) looked convincing or conclusive from a small number of examples or a crude summary '
    'statistic, and were each contradicted by a more careful check &mdash; the framing artifact did not '
    'survive being checked against the full population, and the hair hypothesis&rsquo;s regional-average '
    'proxy missed an effect that a direct crop-and-retrain ablation later confirmed was real. Age '
    '(&sect;4.8) cuts the opposite way: the naive proxy (a simple correlation) was the strongest signal '
    'of the whole report, and it was the <i>controlled retrain</i> that revealed it wasn&rsquo;t what the '
    'model actually relies on &mdash; a reminder that neither a correlation nor a small example set is a '
    'substitute for actually testing the mechanism. A third proposed check (grayscale ablation) was '
    'conceptually wrong and was caught only because a collaborator asked &ldquo;what do I miss?&rdquo; instead of '
    'accepting the plan. We think this is a fair representation of what interpretability work on small '
    'models often looks like in practice: '
    'plausible-looking proxies can point the wrong way in either direction, and checking a hypothesis '
    'with the most direct available test (or getting a second opinion) is not optional.', body))

story.append(section('7. Limitations and Ethical Considerations'))
story.append(P(
    '<b>All 5,500 images, both Caucasian and Asian subsets, were rated by the same pool of 60 Asian '
    'raters</b> [<a href="#ref1" color="blue">1</a>, <a href="#ref4" color="blue">4</a>]. We did not know '
    'this when we framed &sect;4.4&ndash;4.7 as a test of whether beauty judgments transfer across rater '
    'populations; they do not test that. What they test is narrower: whether a model trained on one '
    'photographed-face population&rsquo;s ratings (from a fixed rater culture) generalizes to another '
    'photographed-face population&rsquo;s ratings from the <i>same</i> raters. This does not weaken the '
    'calibration-collapse finding itself, but it means the CF&rarr;AF failure cannot be attributed to, or '
    'used as evidence about, cross-cultural disagreement in aesthetic judgment &mdash; only to the model '
    'failing to generalize across face populations under a fixed rater culture. It also offers an '
    'alternative reading of our unexpectedly strong &sect;4.1 baseline: Perrett et al. (cited in '
    '[<a href="#ref4" color="blue">4</a>]) suggest raters may show higher internal consistency judging '
    'faces from their own population, which would inflate a model&rsquo;s apparent accuracy independent of '
    'anything the model itself does well &mdash; a confound we did not control for.', body))
story.append(P(
    'SCUT-FBP5500 is licensed for non-commercial research only; we did not scrape additional images '
    'and this report does not redistribute the dataset. The &ldquo;pretty vs. average&rdquo; binarization is a '
    'median split within each subset, not a claim about any individual depicted person, and the mean '
    'rating underlying it reflects 60 specific raters, not a population census. We study only two of '
    'the dataset&rsquo;s four race/gender subsets (Caucasian-female, Asian-female); the binary race/gender '
    'categories are the dataset&rsquo;s own encoding, used here as given, not an endorsement of treating '
    'race or gender as binary or as the relevant axes of human variation in attractiveness research '
    'more broadly. All quantitative results in this report come from a single named model architecture '
    '(ResNet18) and a single training recipe on a comparatively small dataset (750&ndash;2000 images per '
    'subset); we make no claim that these findings generalize to other architectures, larger datasets, '
    'or other populations. The result figures in this report, including face thumbnails, are published '
    'in the same public GitHub repository as the code that produced them, a choice made and discussed '
    'explicitly with the corresponding author given the dataset&rsquo;s license terms. The &sect;4.8 age '
    'estimates come from an off-the-shelf model with its own error and bias (typical published mean '
    'absolute error on benchmark datasets is several years); we did not validate it against human-labeled '
    'ages for this specific dataset, so the age-matching in &sect;4.8 is only as good as that model&rsquo;s '
    'accuracy.', body))

story.append(section('8. Conclusion'))
story.append(P(
    'A ResNet18 classifier separates &ldquo;pretty&rdquo; from &ldquo;average&rdquo; Caucasian-female faces with high, '
    'stable accuracy, but calibrates poorly when applied to Asian-female faces despite retaining a '
    'partially useful ranking signal &mdash; and the reverse direction transfers almost perfectly. Framing '
    'artifacts, image-quality confounds, and training-set size explain none of the asymmetry; excluding '
    'hair via a facial-landmark crop explains some of it (chance-level cross-domain accuracy rises to '
    '0.596) but leaves most of the gap (vs. 0.894 in-domain) unaccounted for. Age turned out to be the '
    'strongest correlate with the label of anything we measured (r=&minus;0.494), and AF&rsquo;s narrower, '
    'younger age distribution than CF&rsquo;s is a plausible untested contributor to the asymmetry &mdash; but '
    'an age-matched retrain of the CF-only baseline barely changed its AUC, so this strong pattern in '
    'the human ratings does not appear to be a strong lever in the model itself. The qualitative headline '
    '&mdash; SCUT-FBP5500-trained models generalize poorly outside their training population &mdash; is '
    'already established in the literature [<a href="#ref4" color="blue">4</a>, '
    '<a href="#ref5" color="blue">5</a>]; what we add is a specific, transparently-reported trace of '
    'candidate mechanisms, most of which turned out not to be the answer, plus the previously '
    'unremarked fact that our two &ldquo;populations&rdquo; share a single rater culture (&sect;7), which narrows '
    'what the asymmetry can be taken to show. Adding two more datasets with genuinely independent '
    'rater cultures (&sect;5) confirms a real transferable signal exists &mdash; every '
    'leave-one-source-out direction beats chance &mdash; but does not close the calibration gap: one '
    'direction transfers with identical in-/out-of-domain AUC (0.750), while another collapses to '
    'near-chance out-of-domain accuracy (0.559) despite a respectable AUC (0.676). We report this as '
    'an audit of what a plausible-looking classifier does and does not generalize, not as evidence for '
    'or against the objectivity of beauty and not as a novel contribution to the field, and we hope '
    'the sequence of tested and partly-wrong hypotheses is at least as useful to a reader as the ones '
    'that held up.', body))

story.append(section('Acknowledgments'))
story.append(P(
    'We thank Google Colab&rsquo;s free GPU tier, the maintainers of SCUT-FBP5500, the Chicago Face '
    'Database, and the Face Research Lab London Set for releasing their data for research, and the '
    'corresponding author for catching a wrong claim about grayscale images before it became a wrong '
    'experiment, and for pushing back on Viola-Jones when a real deep-learning face detector was the '
    'right call.', body))

story.append(section('References'))
story.append(P(
    '<a name="ref1"/>[1] L. Liang, L. Lin, L. Jin, D. Xie, and M. Li. SCUT-FBP5500: A Diverse Benchmark '
    'Dataset for Multi-Paradigm Facial Beauty Prediction. <i>ICPR</i>, 2018. arXiv:1801.06345.', ref_style))
story.append(P(
    '<a name="ref2"/>[2] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. '
    'Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. <i>ICCV</i>, 2017.',
    ref_style))
story.append(P(
    '<a name="ref3"/>[3] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image '
    'Recognition. <i>CVPR</i>, 2016.', ref_style))
story.append(P(
    '<a name="ref4"/>[4] N. Fernandes, S. Soares, and J. Arantes. Analyzing VGG-19&rsquo;s Bias in Facial '
    'Beauty Prediction: Preference for Feminine Features. <i>International Journal of Advanced Computer '
    'Science and Applications</i>, 15(10), 2024.', ref_style))
story.append(P(
    '<a name="ref5"/>[5] A. Zejmo, M. Gielert, M. Grabski, and B. Kostek. Assessing the '
    'Attractiveness of Human Face Based on Machine Learning. <i>Procedia Computer Science</i>, '
    '225:1019&ndash;1027, 2023.', ref_style))
story.append(P(
    '<a name="ref6"/>[6] S. I. Serengil and A. Ozpinar. HyperExtended LightFace: A Facial Attribute '
    'Analysis Framework. <i>International Conference on Engineering and Emerging Technologies '
    '(ICEET)</i>, 2021.', ref_style))
story.append(P(
    '<a name="ref7"/>[7] D. S. Ma, J. Correll, and B. Wittenbrink. The Chicago Face Database: A '
    'Free Stimulus Set of Faces and Norming Data. <i>Behavior Research Methods</i>, '
    '47(4):1122&ndash;1135, 2015.', ref_style))
story.append(P(
    '<a name="ref8"/>[8] L. DeBruine and J. Benedict. Face Research Lab London Set. '
    '<i>figshare</i>, dataset, 2017.', ref_style))
story.append(P(
    '<a name="ref9"/>[9] W. Wu, H. Peng, and S. Yu. YuNet: A Tiny Millisecond-level Face Detector. '
    '<i>Machine Intelligence Research</i>, 20(5):656&ndash;665, 2023.', ref_style))

doc = make_doc()
doc.build(story)
print(f'Wrote {OUT_PDF}')
