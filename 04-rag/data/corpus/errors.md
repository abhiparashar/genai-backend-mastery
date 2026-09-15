# Error Code Reference

## Inventory Errors

Error SKU-4471 indicates that the warehouse inventory sync failed for a
specific item. The usual cause is a stale product record in the upstream
catalogue. Resolution: re-run the catalogue sync job, then retry.

Error SKU-8830 means the item was found in the catalogue but has no assigned
warehouse location.

## Payment Errors

Error PAY-9902 means the payment gateway rejected the transaction. This is
almost always an expired card or insufficient funds; it is not a system fault.

Error PAY-1180 indicates a timeout communicating with the payment provider.
Transactions in this state must be reconciled manually before any retry, or
the customer may be charged twice.
