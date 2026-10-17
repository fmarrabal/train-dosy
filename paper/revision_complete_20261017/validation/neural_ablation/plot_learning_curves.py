"""Plot existing histories only; never train or select a checkpoint here."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
SEEDS = [172901, 172902, 172903, 172904, 172905]

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.titlesize': 13, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.0), sharex=True,
                             sharey=True, constrained_layout=True)
    sources = []
    for ax, seed in zip(axes.flat, SEEDS):
        path = BASE / 'training' / f'seed_{seed}.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        history = data['history']
        epochs = [r['epoch'] for r in history]
        training = [r['train_loss'] for r in history]
        validation = [r['validation_loss'] for r in history]
        best = data['best_epoch']
        selected = next(r for r in history if r['epoch'] == best)
        assert len(epochs) == 400 and epochs[-1] == 400
        assert selected['validation_loss'] == min(validation)
        ax.plot(epochs, training, color='#2166ac', lw=1.5)
        ax.plot(epochs, validation, color='#b35806', lw=1.5)
        ax.axvline(best, color='#555555', ls=':', lw=1.1)
        ax.plot(best, selected['validation_loss'], marker='o', ms=5,
                mec='#b35806', mfc='white', mew=1.2, clip_on=False)
        ax.set_title(f'Seed {seed}\nSelected epoch {best}')
        ax.set_yscale('log')
        ax.set_xlim(0, 410)
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.grid(which='major', alpha=.2, lw=.5)
        sources.append({'seed': seed, 'path': str(path.relative_to(BASE)),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'best_epoch': best,
                        'best_validation_loss': selected['validation_loss']})
    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    legend_ax.legend(handles=[
        Line2D([0], [0], color='#2166ac', lw=1.5, label='Training loss'),
        Line2D([0], [0], color='#b35806', lw=1.5, label='Validation loss'),
        Line2D([0], [0], color='#555555', ls=':', lw=1.1,
               label='Selected validation\ncheckpoint')], loc='upper left',
        frameon=False, fontsize=12)
    legend_ax.text(.03, .54, 'Fixed budget: 400 epochs\n64 training / 16 validation\nproblems per seed;\nshared 80-problem bank.\n\nFour selections reach the cap.\nThese curves do not establish\nlearning convergence.',
                   transform=legend_ax.transAxes, va='top', fontsize=12,
                   linespacing=1.35)
    fig.supxlabel('Epoch')
    fig.supylabel('Smooth L1 loss (logarithmic scale)')
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(BASE / f'learning_curves.{extension}', dpi=180)
    plt.close(fig)
    manifest = {'description': 'Existing saved training and validation histories; no new training.',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sources': sources,
                'outputs': {f'learning_curves.{ext}': hashlib.sha256(
                    (BASE / f'learning_curves.{ext}').read_bytes()).hexdigest()
                    for ext in ['pdf', 'svg', 'png']}}
    (BASE / 'learning_curves_manifest.json').write_text(json.dumps(manifest, indent=2))
    (BASE / 'learning_curves_caption.txt').write_text(
        'Training and whole-problem validation Smooth L1 loss for five separately seeded RAI-Net rankers. '
        'Each run uses 64 training and 16 validation problems drawn from the same historical '
        '80-problem bank; feature normalization uses training records only. Dotted lines and '
        'open markers indicate the checkpoint with the lowest observed validation loss, '
        'at epochs 400, 400, 400, 400 and 64. Four selections reach the prespecified '
        '400-epoch budget, so the histories do not establish learning convergence. '
        'The target is signed log internal-validation prediction gain. As recorded by the '
        'unchanged trainer, training loss precedes each Adam update and validation loss follows it. '
        'Test data are absent from training, normalization and checkpoint selection.\n')
    print(json.dumps({'outputs': list(manifest['outputs']),
                      'selected_epochs': [s['best_epoch'] for s in sources]}))

if __name__ == '__main__':
    main()
