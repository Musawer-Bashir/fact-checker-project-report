create extension if not exists "pgcrypto";

create table if not exists claims (
  id uuid primary key default gen_random_uuid(),
  raw_text text not null,
  normalized_text text not null,
  hash text not null,
  created_at timestamptz not null default now()
);

create index if not exists claims_hash_idx on claims (hash, created_at desc);

create table if not exists verdicts (
  id uuid primary key default gen_random_uuid(),
  claim_id uuid not null references claims(id) on delete cascade,
  label text not null check (label in (
    'True', 'Mostly True', 'Mixed', 'Mostly False', 'False',
    'Unverifiable', 'No Evidence'
  )),
  confidence real not null check (confidence >= 0 and confidence <= 1),
  summary text not null,
  reasoning text not null,
  citations jsonb not null default '[]'::jsonb,
  evidence_count int not null default 0,
  model_version text not null,
  created_at timestamptz not null default now()
);

create index if not exists verdicts_claim_id_idx on verdicts (claim_id);

alter table claims enable row level security;
alter table verdicts enable row level security;

create policy "claims readable" on claims for select using (true);
create policy "verdicts readable" on verdicts for select using (true);
