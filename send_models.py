from pathlib import Path
import os
import subprocess
from subprocess import PIPE
from io import StringIO
from smtplib import SMTP
from email.message import EmailMessage

from Bio.PDB import PDBIO, PDBParser
import click


@click.command()
@click.option('--target-id', type=click.Path(exists=True, file_okay=False), required=True, help='Target ID directory')
@click.option('--models', type=click.STRING, required=True, help='e.g.) --models 1,2,4,9')
def main(target_id, models):
    with Path(f'{target_id}/DELTA-BLAST/casp.pdb').open('w') as f:
        f.writelines([
            'PFRMAT TS\n',
            f'TARGET {target_id}\n',
            'AUTHOR {AUTHOR_ID}\n',
            'METHOD ExMachina\n',
        ])
        for i, rank in enumerate(models.split(','), 1):
            if not Path(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.pdb').exists():
                continue
            remark = [_ for _ in Path(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.pdb').read_text().splitlines() if _.startswith('REMARK') and 'TEMPLATE:' in _][0]
            template_id = remark.split()[3]
            template_chain = remark.split()[4].split(':')[1]
            f.writelines([
                f'MODEL  {i}\n',
                f'PARENT {template_id}_{template_chain}\n'
            ])
            pdbio = PDBIO()
            pdbio.set_structure(PDBParser().get_structure(target_id, f'{target_id}/DELTA-BLAST/{rank}/{target_id}.pdb'))
            pdbio.save(f)

    msg = EmailMessage()
    with Path(f'{target_id}/DELTA-BLAST/casp.pdb').open() as f:
        msg.set_content(f.read())
    msg['Subject'] = f'{target_id} by ishidalab'
    msg['From'] = ''
    msg['To'] = Path(f'{target_id}/reply_email').read_text()
    msg['CC'] = ''
    with SMTP('localhost') as s:
        s.send_message(msg)


if __name__ == '__main__':
    main()
