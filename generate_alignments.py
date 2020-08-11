import os
import subprocess
from subprocess import PIPE
from pathlib import Path
from io import StringIO
from smtplib import SMTP
from email.message import EmailMessage
import tarfile
from time import sleep

import requests
import boto3
from slack_webhook import Slack
from Bio import SeqIO, SearchIO, AlignIO
from Bio.Alphabet import generic_protein
from Bio.Blast.Applications import NcbideltablastCommandline
from Bio.PDB import PDBList
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pymol
from PIL import Image
import modeller
from modeller import environ
from modeller.automodel import automodel

from exmachina import _run_all
from machina.generate_models import replace_missing_residues


POLLING_INTERVAL = 3600

SQS_QUEUE_URL = ''
# Dead Letter Queue
# SQS_QUEUE_URL = ''

SLACK_WEBHOOK_URL = ''
SLACK_OAUTH_TOKEN = ''
SLACK_CHANNELS = '#casp14'

DELTA_BLAST_TOP_HITS = 10


class CD:
    def __init__(self, to, mkdir=False):
        self._curr_dir = Path(to).absolute()
        self._prev_dir = Path.cwd().absolute()
        if mkdir:
            self.curr.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        os.chdir(self.curr)
        return self

    def __exit__(self, _, __, ___):
        os.chdir(self.prev)

    @property
    def prev(self):
        return self._prev_dir

    @property
    def curr(self):
        return self._curr_dir


def main():
    sqs = boto3.client('sqs')

    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All']
        )
        if 'Messages' in response:
            break
        sleep(POLLING_INTERVAL)

    print(response)
    message = response['Messages'][0]
    sequence = message['Body']
    target_id = message['MessageAttributes']['target']['StringValue']
    reply_email = message['MessageAttributes']['reply-e-mail']['StringValue']

    Path(target_id).mkdir(exist_ok=True)
    Path(f'{target_id}/reply_email').write_text(reply_email)
    SeqIO.write(
        SeqIO.read(StringIO(sequence), 'fasta', alphabet=generic_protein),
        f'{target_id}/{target_id}.fasta', 'fasta'
    )

    Slack(url=SLACK_WEBHOOK_URL).post(
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Target ID*: {target_id}"}},
            {"type": 'section', 'text': {'type': 'mrkdwn', "text": f"*Reply E-mail*: {reply_email}"}},
            {'type': 'section', 'text': {'type': 'mrkdwn', 'text': f"```\n{sequence}\n```"}}
        ]
    )

    msg = EmailMessage()
    msg.set_content(f'{target_id} - query received by ishidalab')
    msg['Subject'] = f'{target_id} - query received by ishidalab'
    msg['From'] = ''
    msg['To'] = ''
    msg['CC'] = ''
    with SMTP('localhost') as s:
        s.send_message(msg)

    blast_db = 'pdbaa'
    _, __ = NcbideltablastCommandline(
        rpsdb='cdd_delta',
        db=blast_db,
        query=f'{target_id}/{target_id}.fasta',
        evalue=1000,
        outfmt=5,
        out=f'{target_id}/{target_id}.xml',
        num_threads=os.cpu_count()
    )()

    result_set = SearchIO.read(f'{target_id}/{target_id}.xml', 'blast-xml')
    query_record = SeqIO.read(f'{target_id}/{target_id}.fasta', 'fasta', alphabet=generic_protein)

    s = ''
    for rank, hit in enumerate(result_set.hits[:DELTA_BLAST_TOP_HITS], 1):
        res = subprocess.run(['blastdbcmd', '-db', blast_db, '-entry', hit.id],
                             stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
        hit_record = SeqIO.read(StringIO(res.stdout), 'fasta', alphabet=generic_protein)
        if not Path(f'{target_id}/DELTA-BLAST/{rank}/{hit_record.id[:4].upper()}.ent').exists():
            p = PDBList().retrieve_pdb_file(hit_record.id[:4], pdir=f'{target_id}/DELTA-BLAST/{rank}', file_format='pdb')
            if Path(p).exists():
                Path(p).rename(f'{target_id}/DELTA-BLAST/{rank}/{hit_record.id[:4].upper()}.ent')
        hsp = hit.hsps[0]
        s += f'#{rank:2d} {hit.id} Evalue={hsp.evalue}\n  {hit.description}\n  Coverage={hsp.query_span}/{len(query_record)}={hsp.query_span/len(query_record):.2f} Range=Q:{hsp.query_range},T:{hsp.hit_range}\n'

    Slack(url=SLACK_WEBHOOK_URL).post(
        blocks=[
            {'type': 'section', 'text': {'type': 'mrkdwn', 'text': '*Template Search*: DELTA-BLAST'}},
            {'type': 'section', 'text': {'type': 'mrkdwn', 'text': f'```\n{s}\n```'}}
        ]
    )

    if len(result_set.hits) == 0:
        return

    for rank, hit in enumerate(result_set.hits[:DELTA_BLAST_TOP_HITS], 1):
        res = subprocess.run(['blastdbcmd', '-db', blast_db, '-entry', hit.id],
                             stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
        hit_record = SeqIO.read(StringIO(res.stdout), 'fasta', alphabet=generic_protein)
        Path(f'{target_id}/DELTA-BLAST/{rank}').mkdir(exist_ok=True, parents=True)
        hsp = hit.hsps[0]
        s = 0 if hsp.query_start - 10 < 0 else hsp.query_start - 10
        SeqIO.write(query_record[s:s+hsp.query_span+10], f'{target_id}/DELTA-BLAST/{rank}/{target_id}.fasta', 'fasta')
        s = 0 if hsp.hit_start - 10 < 0 else hsp.hit_start - 10
        SeqIO.write(hit_record[s:s+hsp.hit_span+10], f'{target_id}/DELTA-BLAST/{rank}/{hit_record.id}.fasta', 'fasta')
        _run_all(
            out_dir=f'{target_id}/DELTA-BLAST/{rank}',
            query=f'{target_id}/DELTA-BLAST/{rank}/{target_id}.fasta',
            template=f'{target_id}/DELTA-BLAST/{rank}/{hit_record.id}.fasta',
            blastdb='uniref90',
            num_iterations=3,
            num_threads=os.cpu_count(),
            flann_x='scop40_logscore_tmscore0.5_window5_ratio0.1_x.npy',
            flann_y='scop40_logscore_tmscore0.5_window5_ratio0.1_y.npy',
            flann_index='flann19_scop40_logscore_tmscore0.5_window5_ratio0.1',
            num_neighbors=1000,
            score_out_name='score.npy',
            score_matrix='score.npy',
            open_penalty=0.1,
            extend_penalty=0.0001,
            alignment_out_name='alignments.npy'
        )

    modeller.log.minimal()
    for rank, hit in enumerate(result_set.hits[:DELTA_BLAST_TOP_HITS], 1):
        res = subprocess.run(['blastdbcmd', '-db', blast_db, '-entry', hit.id],
                             stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
        hit_record = SeqIO.read(StringIO(res.stdout), 'fasta', alphabet=generic_protein)
        template_id, template_chain = hit_record.id[:4], hit_record.id[5]
        mafft_input = [SeqIO.read(f'{target_id}/{target_id}.fasta', 'fasta')]
        with CD(f'{target_id}/DELTA-BLAST/{rank}'):
            template_file = Path(f'{template_id.upper()}.ent')
            if not template_file.exists():
                continue
            aln = np.load('alignments.npy')
            best = aln[np.argmax([float(_[2]) for _ in aln])]
            target_seq = best[0]
            template_seq = replace_missing_residues(best[1], template_id, template_chain, template_file.as_posix())
            mafft_input.append(SeqRecord(Seq(target_seq, generic_protein), id=target_id, name='', description=''))
            mafft_input.append(SeqRecord(Seq(template_seq, generic_protein), id=template_id, name='', description=''))
            SeqIO.write(mafft_input, 'mafft_input.fasta', 'fasta')
            Path('mafft_table').write_text('1\n2 3')
            res = subprocess.run(['mafft', '--clustalout', '--merge', 'mafft_table', 'mafft_input.fasta'],
                                 stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
            aln = MultipleSeqAlignment(AlignIO.read(StringIO(res.stdout), 'clustal', alphabet=generic_protein)[::2])
            SeqIO.write([
                SeqRecord(aln[0].seq, id=target_id, name='', description=f'sequence:{target_id}::::::::'),
                SeqRecord(aln[1].seq, id=template_id, name='', description=f'structureX:{template_id}::{template_chain}::{template_chain}::::')
            ], 'alignment.pir', 'pir')
            try:
                automodel(environ(), alnfile='alignment.pir', knowns=[template_id], sequence=target_id).make()
            except:
                try:
                    env = environ()
                    env.io.convert_modres = False
                    automodel(env, alnfile='alignment.pir', knowns=[template_id], sequence=target_id).make()
                except:
                    pass
            [f.unlink() for f in Path().glob(f"{target_id}.V*")]
            [f.unlink() for f in Path().glob(f"{target_id}.D*")]
            [f.unlink() for f in Path().glob(f"{target_id}.ini")]
            [f.unlink() for f in Path().glob(f"{target_id}.rsr")]
            [f.unlink() for f in Path().glob(f"{target_id}.sch")]
            [f.rename(f'{target_id}.pdb') for f in Path().glob(f"{target_id}.B*.pdb")]

    model_img = Image.new('RGB', (640*5, 480*int(DELTA_BLAST_TOP_HITS/5)))
    for rank, hit in enumerate(result_set.hits[:DELTA_BLAST_TOP_HITS], 1):
        if not Path(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.pdb').exists():
            continue
        pymol.cmd.load(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.pdb')
        pymol.cmd.png(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.png')
        pymol.cmd.delete('all')
        x = rank - 1
        model_img.paste(
            Image.open(f'{target_id}/DELTA-BLAST/{rank}/{target_id}.png'),
            (640*(x%5), 480*int(x/5))
        )
    model_img.save(f'{target_id}/DELTA-BLAST/{target_id}.png')
    with Path(f'{target_id}/DELTA-BLAST/{target_id}.png').open('rb') as f:
        res = requests.post(
            "https://slack.com/api/files.upload",
            data={"token": SLACK_OAUTH_TOKEN, "channels": SLACK_CHANNELS, "title": "Models"},
            files={'file': f}
        )

    with tarfile.open(f'{target_id}.tar.gz', 'w:gz') as f:
        f.add(f'{target_id}')
    with Path(f'{target_id}.tar.gz').open('rb') as f:
        res = requests.post(
            "https://slack.com/api/files.upload",
            data={"token": SLACK_OAUTH_TOKEN, "channels": SLACK_CHANNELS, "title": "PDB"},
            files={'file': f}
        )

    sqs.delete_message(
        QueueUrl=SQS_QUEUE_URL,
        ReceiptHandle=message['ReceiptHandle']
    )


if __name__ == '__main__':
    while True:
        main()
